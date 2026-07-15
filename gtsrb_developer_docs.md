# Developer Documentation: `gtsrb_vit_runpod_v2.ipynb`

**GTSRB Traffic Sign Classification — ViT-Base Fine-Tuning Pipeline**

This document explains the notebook cell-by-cell, in enough depth that someone with basic Python knowledge but no deep learning background can understand *what* every piece of code does, *why* it's written that way, and *what to change* if they need to adapt it. It also documents real bugs hit during development and how they were diagnosed, since those are usually the most useful part of any pipeline's documentation.

---

## 1. What this notebook does, in one paragraph

It fine-tunes a pretrained Vision Transformer (`ViT-Base`, from the `keras_hub` library) to classify German traffic signs into 43 categories, using the official GTSRB dataset. It downloads and prepares the data itself, trains in two phases (frozen backbone → partially unfrozen fine-tuning), evaluates on the official test set with a verified, correct label mapping, and packages the results for download. It's designed to run on a single rented GPU (RunPod), not a local machine, though most of it would run anywhere with a GPU and internet access.

---

## 2. Prerequisites

| Requirement | Why |
|---|---|
| One NVIDIA GPU, **Ada or Ampere generation or older** (RTX 3090/4090, A5000/A6000/A100, L4, L40S, T4) | See §4 — Blackwell-generation cards (RTX 5090, RTX PRO 4000/4500/5000/6000) are not supported by current TensorFlow builds |
| ≥16GB VRAM recommended | Comfortable headroom for `BATCH_SIZE = 128` at 224×224 resolution |
| Python 3.11+, Jupyter | Notebook environment |
| Internet access to `sid.erda.dk` | Dataset download source |
| ~10GB free disk | Dataset (~350MB) + checkpoints (~350-1000MB each) + working files |

**Packages** (installed once per environment, not baked into the notebook itself):

```bash
pip install --upgrade --break-system-packages keras keras-hub "tensorflow[and-cuda]" pandas numpy scikit-learn seaborn matplotlib pillow tqdm
apt-get update -qq && apt-get install -y -qq unzip wget
```

`--break-system-packages` is needed on Debian/Ubuntu-based images where pip refuses to install into the system Python environment by default (this is standard for most cloud GPU rental images, which don't use virtualenvs).

---

## 3. Directory layout the notebook assumes

```
/workspace/
├── data/
│   ├── gtsrb_raw/              # downloaded .zip archives + their extracted contents
│   └── gtsrb/
│       ├── Train/<class_id>/*.png    # e.g. Train/00007/00007_00012_00003.png
│       ├── Test/*.png
│       └── Test.csv                  # columns: Path, ClassId
└── outputs/
    └── gtsrb_vit/
        ├── phase1_head_best.keras
        ├── phase2_finetuned_best.keras
        ├── gtsrb_vit_final.keras
        ├── training_history.csv
        ├── training_curves.png
        ├── classification_report.txt
        ├── confusion_matrix.png
        └── test_predictions.csv
```

Everything under `/workspace` is assumed to be on **persistent storage** (a RunPod volume disk or network volume) — see §9 for why this matters.

---

## 4. Section 1 — Environment check & GPU smoke test

**Purpose:** fail fast, in seconds, if the GPU can't actually run TensorFlow — rather than discovering this after a 10-minute dataset download.

```python
gpus = tf.config.list_physical_devices("GPU")
details = tf.config.experimental.get_device_details(gpus[0])
cc = details.get("compute_capability")

if cc is not None and cc[0] >= 12:
    print("*** WARNING: Blackwell-class GPU (sm_120)... ***")

with tf.device("/GPU:0"):
    a = tf.random.normal((512, 512))
    b = tf.random.normal((512, 512))
    c = tf.matmul(a, b)
    result = float(tf.reduce_sum(c).numpy())
```

**Why this exists (real incident):** during development, an RTX PRO 4500 (a "Blackwell"-generation GPU, compute capability `12.0`) was used. Every single GPU operation failed with `CUDA_ERROR_INVALID_PTX` — not an out-of-memory error, not a driver issue, but a fundamental gap: the TensorFlow build had no precompiled kernels for compute capability 12.x, and its just-in-time PTX fallback compilation was also incompatible. This is a known, unresolved issue across TensorFlow 2.18–2.21 as of this writing, not a local misconfiguration.

**What the check does:** reads the GPU's compute capability directly from TensorFlow's device details, and if it's ≥12 (Blackwell), prints an explicit warning *before* attempting a real computation. Then it runs an actual small matrix multiplication — this is the part that would genuinely fail on an incompatible GPU, giving a fast, clear failure instead of a cryptic error 40 minutes into a training run.

**If you hit this:** don't try to fix the CUDA install — redeploy on an older-generation GPU (Ada or Ampere).

---

## 5. Section 2 — Paths & hyperparameters

All configuration lives in one cell so nothing is buried inside logic later on.

```python
BATCH_SIZE = 128
VAL_FRACTION = 0.2

HEAD_EPOCHS = 8
HEAD_LR = 1e-3

FINE_TUNE_EPOCHS = 15
FINE_TUNE_LR = 1e-5
UNFREEZE_LAST_N_BLOCKS = 4
```

**Key parameters explained:**

- `BATCH_SIZE = 128` — how many images are processed together per training step. Larger batches use more GPU memory but train faster per-epoch (better GPU utilization) and give smoother gradient estimates. If you hit an out-of-memory error, this is the first thing to reduce (try 64).
- `HEAD_LR = 1e-3` vs `FINE_TUNE_LR = 1e-5` — a **100x difference in learning rate** between the two phases. This is deliberate and important: Phase 1 trains a randomly-initialized classification head from scratch, which tolerates large update steps. Phase 2 fine-tunes pretrained transformer weights that already contain useful knowledge — large updates there would destructively overwrite that knowledge rather than gently adapting it. Using too high a learning rate in Phase 2 is one of the most common ways this kind of fine-tuning silently fails to improve, or actively gets worse.
- `UNFREEZE_LAST_N_BLOCKS = 4` — of ViT-Base's 12 transformer blocks, only the last 4 are made trainable in Phase 2. Later blocks encode more task-specific, abstract features; earlier blocks encode generic low-level patterns (edges, textures) that transfer well without modification. This is a common, sensible default — increasing it (e.g. to 8) lets the model adapt more, at the cost of more compute and higher overfitting risk given the dataset's size.
- `USE_MIXED_PRECISION = False` — left off by default. Mixed precision (using 16-bit floats for most computation) roughly halves memory use and can meaningfully speed up training on Ada-generation GPUs, but occasionally interacts badly with `from_logits=True` losses via numerical instability. It's flagged as worth trying once you have one confirmed-working run, not before.

---

## 6. Section 3 — Dataset download

```python
BASE_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370"
ARCHIVES = ["GTSRB_Final_Training_Images.zip", "GTSRB_Final_Test_Images.zip", "GTSRB_Final_Test_GT.zip"]

for name in ARCHIVES:
    dest = os.path.join(RAW_DIR, name)
    if os.path.exists(dest):
        print(f"Already downloaded: {name}")
        continue
    !wget -c -q --show-progress -O {dest} {BASE_URL}/{name}
```

**Why this exact source, and not Kaggle or Hugging Face:** two other sources were tried and rejected during development —

1. **Kaggle's dataset API** — failed because Kaggle's authentication token format had changed (new-style `KGAT_...` API tokens aren't the classic `{"username":..., "key":...}` JSON `kaggle.json` format the `kaggle` CLI historically expects).
2. **Hugging Face's rehosted copy** (`bazyl/GTSRB`) — downloaded fine, but its parquet-based storage **discarded the original filenames**, replacing images with anonymous indices. This silently broke the track-aware split (§8) — GTSRB's track structure is only visible in filenames like `00000_00029.ppm`.

The official mirror (hosted by University of Copenhagen, the same source `torchvision.datasets.GTSRB` uses) preserves original filenames and needs no authentication at all.

**Idempotency:** the `if os.path.exists(dest): continue` check means re-running this cell after the files are already downloaded is a fast no-op — this is what "Sections 3-4 are one-time" means in the notebook's intro. `wget -c` additionally resumes an interrupted download rather than restarting it.

---

## 7. Section 4 — Format conversion (PPM → PNG)

The official archives ship images in `.ppm` format, which TensorFlow's image decoders don't support natively. This section converts every image to `.png` **once**, using a multiprocessing pool across all available CPU cores, while explicitly preserving the original filename stem:

```python
dst_name = os.path.splitext(fname)[0] + ".png"   # keeps 00000_00029 stem
```

This one line is the crux of why the official mirror was chosen — the track ID embedded in the filename survives the format conversion unchanged.

This section also rebuilds `Test.csv` from the official ground-truth file (`GT-final_test.csv`, semicolon-separated), translating filenames to match the converted `.png` names:

```python
gt = pd.read_csv(gt_path, sep=";")
test_df_out = pd.DataFrame({
    "Path": ["Test/" + os.path.splitext(f)[0] + ".png" for f in gt["Filename"]],
    "ClassId": gt["ClassId"].astype(int),
})
```

---

## 8. Section 5 — The track-aware stratified split

This is the single most important piece of methodology in the whole notebook, so it's worth understanding thoroughly.

**The problem:** GTSRB's training images are frames extracted from video — each physical sign produces ~30 near-duplicate consecutive frames, called a *track*. A naive random `train_test_split` on individual images will scatter frames of the same physical sign across both training and validation sets. The model then gets evaluated on images that are, for practical purposes, near-copies of things it already trained on — producing a validation accuracy that looks great but doesn't reflect real generalization.

**Step 1 — recover track IDs from filenames:**

```python
def list_gtsrb_train_files(train_dir):
    for class_dir in sorted(os.listdir(train_dir)):
        for fname in os.listdir(class_path):
            m = re.match(r"(\d+)_(\d+)", fname)
            if m is None:
                raise ValueError(f"Filename '{fname}' has no track structure...")
            rows.append({
                "filepath": ...,
                "class_id": int(class_dir),
                "track_id": f"{class_dir}_{m.group(1)}",
            })
```

Note the `raise ValueError` if a filename doesn't match the expected pattern — this is a deliberate **fail-loud** design choice. If someone later swaps in a differently-formatted dataset, this stops execution immediately with an explanatory message, rather than silently producing a meaningless split.

**Step 2 — split by track, stratified per class:**

```python
def stratified_track_split(df, val_frac, seed):
    rng = np.random.RandomState(seed)
    val_tracks = set()
    for class_id, grp in df.groupby("class_id"):
        tracks = np.array(sorted(grp["track_id"].unique()))
        tracks = tracks[rng.permutation(len(tracks))]
        k = max(1, int(round(len(tracks) * val_frac)))
        k = min(k, len(tracks) - 1)          # never take every track of a class
        val_tracks.update(tracks[:k])
    return val_tracks
```

Two design details worth calling out:

- **Splitting happens *within* each class** (`df.groupby("class_id")`), not across the whole dataset at once. An earlier, simpler version that shuffled all tracks globally caused some rare classes (as few as 7-8 tracks) to lose *all* their tracks to the validation side by chance, leaving zero training examples for that class. Stratifying per class guarantees every class is represented on both sides.
- **`k = min(k, len(tracks) - 1)`** guards against the opposite failure — for a class with very few tracks, taking a naive 20% could round up to *all* of them, leaving nothing to train on. This line guarantees at least one track always stays in training.

**Step 3 — verify, don't assume:**

```python
leaked = set(train_df["track_id"]) & set(val_df["track_id"])
assert not leaked, f"Track leakage detected: {list(leaked)[:5]}"
assert train_df["class_id"].nunique() == NUM_CLASSES, "Some classes missing from train!"
assert val_df["class_id"].nunique() == NUM_CLASSES, "Some classes missing from val!"
```

These three assertions are cheap (milliseconds) and catch exactly the two failure modes described above, immediately, rather than discovering a broken split only after a full training run produces suspicious numbers.

---

## 9. Section 6 — Input pipeline (`tf.data`)

```python
def make_dataset(df, training, cache_name):
    ds = tf.data.Dataset.from_tensor_slices((df["filepath"].values, df["class_id"].values.astype("int32")))
    ds = ds.map(decode_image, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    if training:
        ds = ds.shuffle(len(df), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(resize_image, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    return ds
```

**Pipeline order matters here, and it's deliberate:**

1. `decode_image` — reads and decodes the raw PNG at its **original, small resolution** (GTSRB images range from 15×15 to 250×250 pixels).
2. `.cache()` — caches the *decoded, original-size* images in RAM. This is placed *before* resizing on purpose: caching after resizing to 224×224 would need ~23GB of RAM for the full dataset; caching the small originals costs only ~100MB. The tradeoff is that resizing still happens on every epoch — but resizing is cheap compared to disk I/O and PNG decoding, so this ordering gets most of the caching benefit for a fraction of the memory cost.
3. `.shuffle()` — only applied to the training set, and only *after* caching (shuffling before caching would defeat the cache's purpose, since cache order would then depend on shuffle order).
4. `resize_image` — resizes to the fixed `224×224` the ViT preset requires. Output stays as float32 in the original `0-255` range — this matters for Section 7 below.
5. `.batch()` — groups into batches for training.

`AUTOTUNE` (`tf.data.AUTOTUNE`) tells TensorFlow to dynamically choose the number of parallel calls for map operations based on available CPU resources, rather than hardcoding a thread count.

---

## 10. Section 7 — Data augmentation, and a real Keras API bug

```python
class RandomGaussianNoise(layers.Layer):
    """GaussianNoise without Keras's built-in stddev<=1 restriction --
    our images are on a 0-255 scale, not the [0,1] scale that restriction assumes."""
    def __init__(self, stddev, **kwargs):
        super().__init__(**kwargs)
        self.stddev = stddev

    def call(self, inputs, training=None):
        if training:
            noise = tf.random.normal(tf.shape(inputs), mean=0.0, stddev=self.stddev)
            return inputs + noise
        return inputs
```

**Why this custom layer exists, instead of `keras.layers.GaussianNoise`:** Keras's built-in `GaussianNoise` layer hard-validates its `stddev` argument in the constructor:

```python
if not 0 <= stddev <= 1:
    raise ValueError(f"Invalid value received for argument `stddev`...")
```

This implicitly assumes input images are already normalized to a `[0,1]` float range. This pipeline deliberately keeps images on their natural `0-255` scale (matching what `keras_hub`'s ViT preset expects as input — see §11), so a meaningful noise perturbation (`stddev=5.0`, roughly 2% of the value range) is *outside* what the built-in layer will accept at all, regardless of whether it's semantically correct for the data. The fix bypasses the built-in layer entirely with a five-line custom layer that adds Gaussian noise directly via `tf.random.normal`, with no artificial range restriction.

**General lesson for anyone extending this notebook:** built-in preprocessing/augmentation layers in Keras often carry implicit assumptions about input value ranges. Always check what scale a layer expects before assuming a "reasonable-looking" parameter value will behave as intended.

The rest of the augmentation stack is a standard `keras.Sequential` of built-in layers (`RandomBrightness`, `RandomContrast`, `RandomRotation`, `RandomZoom`), applied only to the training pipeline, and only with `training=True` explicitly passed so stochastic layers are actually active:

```python
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
```

---

## 11. Section 8 — Phase 1: frozen-backbone training

```python
model = keras_hub.models.ImageClassifier.from_preset("vit_base_patch16_224_imagenet", num_classes=NUM_CLASSES)
model.backbone.trainable = False
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=HEAD_LR),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
```

`keras_hub.models.ImageClassifier.from_preset(...)` downloads (or loads from local cache) the pretrained `ViT-Base` weights and attaches a fresh, randomly-initialized classification head sized to `num_classes=43`. `model.backbone.trainable = False` freezes every layer in the pretrained body — only the new head's weights will update during Phase 1's `model.fit(...)` call.

`SparseCategoricalCrossentropy(from_logits=True)` — standard multi-class classification loss. `from_logits=True` means the model's raw output layer does **not** apply a softmax internally; the loss function applies it as part of its own (more numerically stable) computation. If you ever see wildly wrong-looking loss values, check whether this flag matches whether your model's final layer includes an activation or not — mismatches here are a common, quiet source of broken training.

Checkpointing:

```python
callbacks_phase1 = [
    keras.callbacks.ModelCheckpoint(phase1_ckpt, monitor="val_accuracy", save_best_only=True, verbose=1),
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True, verbose=1),
]
```

`save_best_only=True` means the checkpoint file always holds the *best-so-far* epoch's weights, not simply the most recent. `restore_best_weights=True` on `EarlyStopping` means that once training stops (either by hitting `HEAD_EPOCHS` or by early stopping triggering), the in-memory `model` object is automatically reset to its best epoch — not left at whatever the final epoch happened to produce.

---

## 12. Section 9 — Phase 2: partial unfreeze and fine-tune

This section had the most debugging iterations during development, so it's documented in the most depth.

### 12.1 Finding the transformer blocks

```python
vit_encoder = model.backbone.get_layer("vit_encoder")
blocks = vit_encoder.encoder_layers
```

**Why not something like `model.backbone.encoder.layers`, which seems more obvious?** Two separate, non-obvious facts about `keras_hub`'s `ViTBackbone` implementation, confirmed by reading the library's actual source code:

1. `ViTBackbone` is built using Keras's **Functional API** (`super().__init__(inputs=inputs, outputs=output)`), not as a subclass with named attributes. This means its internal named sub-layers (`vit_patching_and_embedding`, `vit_encoder`) are **not** accessible as Python attributes (`model.backbone.vit_encoder` raises an `AttributeError` / silently returns `None` via `getattr`). They're only reachable via `model.backbone.get_layer("name")`.
2. The layer holding the actual transformer blocks, `ViTEncoder`, is a plain `keras.layers.Layer`, **not** a `keras.Model` — so it has no built-in `.layers` property (that's a `Model`-only convenience). Its 12 blocks are instead stored in a manually-created list attribute, `self.encoder_layers`, assigned inside the layer's `build()` method, with each block named `transformer_block_{i+1}`.

An earlier version of this code tried to auto-discover the blocks by walking `.layers` attributes recursively — this failed silently (`Found 0 transformer blocks`) precisely because of fact #2 above. **If you upgrade `keras_hub` and this cell starts raising `RuntimeError: Transformer blocks not found`, the library's internal structure has likely changed again — read the printed `model.backbone.summary()` output, and if needed, read the installed `keras_hub` package's source directly** (`site-packages/keras_hub/src/models/vit/vit_layers.py`) rather than guessing.

### 12.2 The freeze/unfreeze logic

```python
before = sum(int(np.prod(w.shape)) for w in model.trainable_weights)

model.backbone.trainable = True   # (1)

n_unfreeze = min(UNFREEZE_LAST_N_BLOCKS, len(blocks))
for b in blocks[:-n_unfreeze]:
    b.trainable = False            # (2)
for b in blocks[-n_unfreeze:]:
    b.trainable = True

for layer in model.backbone.layers:
    if any(k in layer.name.lower() for k in ("embedding", "patch")):
        layer.trainable = False    # (3)
```

The comment on line (1) explains a genuinely non-obvious Keras behavior: **a container layer with `trainable = False` reports zero trainable weights regardless of what its children are individually set to.** You cannot go directly from "everything frozen" to "just unfreeze block 9" — you have to first set the *container* (`model.backbone`) trainable, then selectively re-freeze the specific blocks you don't want trained. Getting this order backwards is a common, silent way to end up training nothing at all while believing you've configured a partial unfreeze.

Line (3) explicitly keeps the patch and position embedding layers frozen even though the backbone container is now trainable overall — these are treated as generic, low-level, and not worth the extra training cost/overfitting risk of adapting them.

### 12.3 The verification assertion — don't trust silent success

```python
after = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
...
assert after > before, (
    "Trainable parameter count did not increase — the unfreeze silently failed. "
    "Do not train; investigate the block names printed above first."
)
```

This assertion exists because of the exact bug described in §12.1: an unfreeze operation that silently does nothing looks *identical* to a correctly-configured one right up until you notice, epochs later, that validation accuracy isn't moving. Counting trainable parameters before and after, and asserting the count actually changed, converts a silent, slow-to-detect failure into an immediate, loud one.

---

## 13. Section 11 — Evaluation, and the label-mapping bug

```python
class_folders = sorted(d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d)))
label_mapping = {int(folder): idx for idx, folder in enumerate(class_folders)}
inverse_mapping = {idx: int(folder) for idx, folder in enumerate(class_folders)}
print("Identity mapping?", all(k == v for k, v in label_mapping.items()))
```

**The bug this prevents:** when Keras assigns integer label indices to class folders (implicitly, wherever folder names get sorted), it sorts them **alphabetically as strings**, not numerically. With unpadded folder names (`"0", "1", "10", "11", "2", ...`), `"10"` sorts before `"2"` — meaning the model's internal index for "class 2" would not equal the integer `2`. Confusing the two produces evaluation numbers that are numerically valid but semantically meaningless (an early, unfixed version of this exact bug produced a nonsensical 3.74% "accuracy" on an otherwise reasonably-trained model).

This notebook sidesteps the ambiguity by building `label_mapping` and `inverse_mapping` **explicitly** from the same class-folder listing used everywhere else, rather than relying on any implicit ordering assumption. Because the official GTSRB mirror uses zero-padded folder names (`"00000"`...`"00042"`), string sort and numeric sort happen to agree here — hence `print("Identity mapping?", ...)` should print `True`. That print statement is a deliberate, cheap sanity check: if it ever prints `False` (e.g. if someone swaps in a differently-named dataset), you know immediately, rather than needing to discover it via a suspiciously bad accuracy number later.

```python
raw_logits = model.predict(test_ds, verbose=1)
y_pred = np.argmax(raw_logits, axis=1)
```

`model.predict` returns raw, unnormalized logits (consistent with `from_logits=True` in the loss function, §11). `np.argmax` picks the index of the largest logit per image — since softmax is monotonic, you don't need to actually apply it to determine the predicted class, only to get calibrated confidence percentages (which Section 12 does separately via `tf.nn.softmax`).

---

## 14. Section 13 — Packaging results for download

```python
zip_path = shutil.make_archive("/workspace/gtsrb_vit_results", "zip", OUTPUT_DIR)
```

This exists because of a real operational issue: everything the notebook produces lives on the pod's storage, which is inaccessible the moment the pod stops being reachable (network issues, accidental termination, etc.). Zipping and downloading is a manual step the notebook reminds you to do — it does **not** happen automatically, and forgetting it before shutting down a pod means losing all trained weights and results.

The notebook also documents `scp` as the recommended transfer method over the Jupyter browser download button, based on a real, repeated finding during development: browser-based downloads through a cloud GPU provider's web proxy layer are frequently slow or fail outright for large files (500MB+), while direct `scp` (bypassing that proxy) is markedly more reliable.

---

## 15. Common errors and their real causes

This table is built from actual errors hit during development, not hypothetical ones.

| Error | Real cause | Fix |
|---|---|---|
| `CUDA_ERROR_INVALID_PTX` on any GPU op | GPU is Blackwell-generation (compute capability ≥12); TensorFlow has no kernels for it | Redeploy on an Ada/Ampere GPU. Don't try to fix the CUDA install. |
| `ValueError: Invalid value received for argument stddev` in augmentation | Built-in `GaussianNoise` hard-enforces `stddev` in `[0,1]`, assuming normalized input | Use the custom `RandomGaussianNoise` layer in §10 instead |
| `Found 0 transformer blocks` / `RuntimeError` in Phase 2 | `ViTBackbone` uses the Functional API (no attribute access); `ViTEncoder` is a plain `Layer` with no `.layers` property | Use `model.backbone.get_layer("vit_encoder").encoder_layers`, not attribute-guessing |
| Trainable parameter count doesn't increase after "unfreezing" | Backbone container `trainable=False` masks all children regardless of their own flag | Set `model.backbone.trainable = True` *before* selectively re-freezing individual blocks |
| Nonsensical low accuracy (e.g. ~4%) despite reasonable training curves | Class folders sorted as strings, not integers, scrambling the label mapping | Always build an explicit `label_mapping` dict; verify it against ground truth before trusting evaluation |
| `FileNotFoundError` when listing `Train/` after using a Hugging Face-sourced dataset | HF's parquet-based rehosting discarded original filenames, breaking track-ID parsing entirely | Use the official GTSRB mirror (§6), which preserves filenames |
| `UserWarning: Skipping variable loading for optimizer...` on `keras.models.load_model(...)` | Expected, harmless — optimizer state (momentum, etc.) isn't restored on reload, only weights. Fine when you're about to `compile()` a fresh optimizer anyway | No action needed if you're recompiling after loading |
| Pod's GPU unavailable on restart | Stopping a pod releases its specific physical GPU; someone else may claim it before you restart | Use RunPod's "automatically migrate" option, or redeploy fresh on a network volume |
| Downloaded dataset extracted with `Train`/`Test` under one level of Kaggle-style unpadded folder names instead of expected structure | Different dataset sources package GTSRB with different folder-naming conventions | Verify with `os.listdir(TRAIN_DIR)` before assuming structure; the explicit label-mapping step (§13) protects against most consequences either way |

---

## 16. How to extend this notebook

**Resume training for more epochs** (useful if the training curve shows validation accuracy still improving when training stops):

```python
model = keras.models.load_model(os.path.join(OUTPUT_DIR, "gtsrb_vit_final.keras"))
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
# then call model.fit(train_ds, validation_data=val_ds, epochs=..., callbacks=[...]) again
```

Requires Sections 1-6 to have been re-run first (to rebuild `train_ds`/`val_ds` if starting a fresh kernel session).

**Try a different amount of unfreezing:** change `UNFREEZE_LAST_N_BLOCKS` in Section 2, then re-run from Section 8 onward.

**Swap in a smaller/larger ViT preset:** change the string in `keras_hub.models.ImageClassifier.from_preset("vit_base_patch16_224_imagenet", ...)` in Section 8 — e.g. to a ViT-Small or ViT-Large preset name, if comparing model capacity is part of your goal. Everything else in the pipeline is preset-agnostic.

**Add class-weighted loss** (if certain classes are underperforming due to low sample counts): compute per-class weights from `train_df["class_id"].value_counts()` and pass a `class_weight` dict into `model.fit(...)`.

---

## 17. Glossary of code-level terms

| Term | Meaning in this notebook |
|---|---|
| `AUTOTUNE` | `tf.data.AUTOTUNE` — lets TensorFlow choose parallelism levels automatically |
| `logits` | A model's raw, un-normalized output scores before softmax is applied |
| `checkpoint` | A saved snapshot of model weights, written via `ModelCheckpoint` |
| `callback` | A Keras object that runs custom logic at points during `model.fit()` (e.g. after each epoch) |
| `trainable` | A boolean flag on a layer controlling whether its weights update during training |
| `track_id` | This notebook's identifier for "which physical sign a training image came from," parsed from the filename |
| `stratified split` | Splitting data so a category (class or track) is proportionally represented in each resulting subset, rather than split purely at random |
| `from_logits=True` | Tells a loss function the model's output has no softmax applied yet, so the loss applies it internally |
