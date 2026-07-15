# The GTSRB Traffic Sign Classifier: A Plain-English Guide

### What this document is

This is a guide to everything behind one dissertation experiment — a computer vision model that learns to recognise German traffic signs. It's written for someone with no machine learning background who wants to actually *understand* what was built and why, not just read a number at the end.

It's split into two halves:
- **Part 1 — The Theory**: the concepts you need, explained with plain-language analogies before the jargon.
- **Part 2 — The Journey**: what we actually did, in order, including the mistakes, the debugging, and why each fix mattered. This is a real case study, not a cleaned-up success story.

There's a glossary at the end for quick lookups.

---

## Part 1 — The Theory

### 1.1 What is the task?

The goal: show the computer a photo of a German road sign, and have it correctly say which of **43 official sign types** it is — stop sign, 50 km/h speed limit, yield, pedestrian crossing, and so on.

This is called **image classification**: the model looks at a picture and picks one label from a fixed list of possibilities. It's one of the oldest and most well-understood problems in AI, which is exactly why it's a great dissertation topic — there's a huge body of prior work to compare against, and clear, objective numbers to measure success.

The dataset is called **GTSRB** (German Traffic Sign Recognition Benchmark) — about 50,000 real photos of signs, taken from cameras mounted in moving cars, split into roughly 39,000 for training and 12,630 for the final test.

### 1.2 Transfer learning — "hire someone who can already read"

Training an image-recognition model completely from scratch requires millions of labelled images and enormous computing power — far more than a dissertation project has access to.

The workaround is **transfer learning**. Instead of teaching a model to understand images from zero, you start with a model that's *already* very good at recognising general visual patterns — edges, shapes, textures, objects — because it was trained on millions of everyday photos (cats, cars, furniture, whatever). You then specialise that existing skill for your specific task.

The analogy: instead of teaching someone to read from the alphabet up, you hire someone who's already fluent, and just train them on your specific paperwork. Much faster, and it works with far less data.

The pretrained model we used is called **ViT-Base**, pretrained on a huge general-purpose image dataset called ImageNet.

### 1.3 What is a Vision Transformer (ViT)?

"Transformer" is the same underlying architecture that powers large language models like ChatGPT — but instead of reading words, a Vision Transformer reads *images*.

Here's roughly how it works:

1. **Cut the image into patches.** A 224×224 pixel image gets sliced into a grid of small square patches (16×16 pixels each, in our case).
2. **Turn each patch into a list of numbers.** Each patch is converted into a vector — a long list of numbers representing its visual content. This is called an *embedding*.
3. **Let the patches "look at" each other.** This is the transformer's signature trick, called **attention**. Every patch compares itself to every other patch in the image and decides how relevant each one is to understanding it. This lets the model notice, for instance, that a red circular border patch on one side of the image is related to a black digit patch elsewhere — together they form a speed limit sign.
4. **Stack many layers of this.** Our model has 12 of these "attention + processing" layers stacked on top of each other, called **transformer blocks**. Early layers tend to pick up simple things (edges, colours); later layers combine those into more abstract concepts (this is a sign, this is a digit, this is the shape of an arrow).
5. **Make a final decision.** After all the layers, the model combines everything it's learned about the image into one final answer: which of the 43 classes is this?

"ViT-Base" specifically means: 12 transformer blocks, and roughly 86 million adjustable numbers (called *parameters*) inside it that get tuned during training.

### 1.4 Freezing and unfreezing — why it matters enormously

When you take a pretrained model and adapt it to a new task, you have a choice for each part of the model:

- **Frozen**: this part keeps exactly the knowledge it already had. It doesn't change during your training.
- **Unfrozen** (a.k.a. **fine-tuned**): this part is allowed to adjust its internal numbers based on your new data.

A common cheap approach: freeze the entire pretrained model, and only train a small new "decision-maker" layer bolted onto the end. This is fast and needs little data — but it means the model is trying to solve your task using knowledge tuned for *general* photos, without ever specialising.

**This distinction was the single biggest lever in this whole project.** More on that in Part 2.

### 1.5 Data splits, and the sneaky problem of "leakage"

To train and honestly evaluate a model, you divide your data into (at least) three piles:

- **Training set**: what the model actually learns from.
- **Validation set**: a held-back set you check the model against *during* training, to see how it's doing on data it hasn't memorised.
- **Test set**: the final, untouched exam — only used once, at the very end, to report your real result.

The entire point of a validation/test set is that the model has **never seen it before**. If it has — even partially — your accuracy number becomes fiction. This is called **data leakage**.

Leakage is often subtle. In this project, it came from something you wouldn't expect: GTSRB's training photos are pulled from *video*, so each physical sign has roughly 30 near-identical consecutive frames (called a **track**). If you split the images randomly, some frames of the *same physical sign* end up in training and others in validation. The model isn't being tested on a genuinely new sign — it's being tested on a sign it basically already saw, just from a slightly different video frame. This inflates your apparent accuracy without meaning anything real.

The fix is a **track-aware split**: keep every frame of a given physical sign entirely on one side of the split. This guarantees the validation set is made of genuinely unseen signs.

### 1.6 How do you measure "how good" a model is?

**Accuracy** — the simplest metric: what percentage of images did the model get right? Easy to understand, but can be misleading if some classes have far more examples than others.

**Precision** — of all the times the model *guessed* a particular class, what fraction were actually correct? Low precision means the model cries wolf on that class too often.

**Recall** — of all the images that *truly were* a particular class, what fraction did the model actually catch? Low recall means the model is missing that class a lot.

**F1-score** — a single number that balances precision and recall together (their harmonic mean). Useful when you want one number that punishes a model for being lopsided (e.g. very precise but missing most examples).

**Confusion matrix** — a grid showing, for every true class, what the model actually predicted. The diagonal (top-left to bottom-right) represents correct predictions; anything off the diagonal is a specific type of mistake. This is by far the most informative tool for understanding *how* a model fails, not just *how often*.

### 1.7 GPUs, and why the specific chip matters

Training a neural network involves enormous numbers of repeated matrix multiplications. A **GPU** (Graphics Processing Unit) is built to do exactly that kind of math in parallel, thousands of times faster than a regular processor (CPU).

Not all GPUs are interchangeable for this purpose, though, for two separate reasons:

- **VRAM (video memory)**: the model and the batch of images being processed have to fit in the GPU's own memory. Too little, and training crashes with an "out of memory" error.
- **Software support**: the *software layer* (in our case, a library called TensorFlow) needs to actually contain pre-built instructions for your exact GPU's internal architecture ("compute capability"). Newer GPU generations sometimes ship before the software catches up — which is exactly what happened to us (see Part 2).

---

## Part 2 — The Journey

This section tells the story in the order it actually happened, including the wrong turns, because the wrong turns are where most of the learning is.

### 2.1 The baseline attempt (Kaggle, dual T4 GPUs)

The first version of the model used the frozen-backbone approach: pretrained ViT-Base, only the small final classification layer trained, on a random (not track-aware) train/validation split.

**Result: 71% accuracy on the official test set — despite showing 88% accuracy on the validation set during training.**

That 17-point gap between validation and test was the red flag. Two separate problems were compounding:

1. **The backbone was frozen.** The model was trying to distinguish fine details of German traffic signs using knowledge tuned on everyday photos, without ever adapting.
2. **The validation split leaked.** Random splitting let near-duplicate frames of the same physical sign land on both sides, making the validation score an inflated, dishonest preview of real performance.

There was also a separate, purely mechanical bug found along the way: GTSRB's class folders are numbered as *strings* (`"0", "1", "10", "11"...`), and when sorted alphabetically rather than numerically, `"10"` comes before `"2"`. If you don't explicitly account for this, the model's internal class numbering can get scrambled relative to the true labels — which, in one early evaluation attempt, produced a nonsensical 3.74% "accuracy" purely from mislabelling, not model failure. The lesson: **always explicitly build and verify your label mapping — never assume folder order matches label order.**

### 2.2 Moving to RunPod, and a run of infrastructure lessons

To get faster iteration than the free Kaggle GPUs, the project moved to **RunPod**, a service that rents out cloud GPUs by the hour. This introduced its own, entirely separate category of lessons — arguably as valuable as the modelling lessons.

**Lesson: not all "top-tier" GPUs work with your software.** The newest GPU generation available (codenamed "Blackwell" — RTX 5090, RTX PRO 4000/4500/5000/6000) turned out to have **no working support in TensorFlow at all**. Every GPU operation failed with a cryptic `CUDA_ERROR_INVALID_PTX` error. This wasn't a configuration mistake — it's a genuine, unresolved compatibility gap between brand-new hardware and the software ecosystem. The fix was simply choosing an older-but-fully-supported GPU generation (Ada or Ampere architecture — RTX 4090, RTX 3090, L4, L40S, A100, A6000, etc.). A small GPU "smoke test" (a tiny real calculation run at the very start of the notebook) was added specifically to catch this in seconds rather than after 45 minutes of setup.

**Lesson: storage types have very different persistence guarantees.** RunPod offers a default **volume disk**, which is deleted the moment you terminate a pod, and an optional **network volume**, which survives regardless of what happens to any individual pod. After nearly losing a fully-prepared dataset to an accidental termination, the project switched to a network volume — cheaper per GB and immune to that whole class of mistake.

**Lesson: a stopped pod can "migrate."** If you stop a pod and someone else rents the exact physical GPU machine it was on, restarting doesn't just fail — RunPod offers to automatically copy your data to a brand-new machine ("pod migration"). This is a real, safe recovery mechanism, but it changes your pod's ID and IP address, which is worth knowing so it doesn't look like something has gone wrong.

**Lesson: downloading large files off a cloud GPU is its own problem.** The final results bundle, including a full ~1GB model checkpoint, kept failing to download through the browser. The fix was two-fold: only keep the files you actually need (dropping redundant intermediate checkpoints cut the download from 3.2GB to ~1GB), and use a direct file-transfer protocol (`scp`) instead of the browser, since the browser's connection is routed through a proxy layer that's a well-documented bottleneck across this entire category of GPU rental service.

### 2.3 Getting the dataset itself was harder than expected

Three separate attempts were needed before the dataset was usable:

1. **Kaggle's download API** — blocked by an authentication token format mismatch (Kaggle had changed how it issues credentials).
2. **Hugging Face's copy of the dataset** — downloaded successfully, but critically, it had **stripped the original filenames**, replacing them with anonymous internal identifiers. This mattered enormously: those filenames were the *only* way to identify which photos belonged to the same physical-sign "track" — without them, a track-aware split (see §1.5) was impossible to build correctly.
3. **The official GTSRB mirror** — downloaded the original archive files directly, which preserved the real filenames (`00000_00029.ppm` = sign track 0, frame 29) and finally made a genuine track-aware split possible.

**The general lesson**: when a dataset gets rehosted or reformatted by a third party, always check whether something silently gets lost in translation — filenames, folder structure, or metadata that later steps quietly depend on.

### 2.4 Building a genuinely fair evaluation

With the real filenames recovered, the split logic was rebuilt to guarantee two things simultaneously:
- No physical sign's frames ever appear on both sides of the split (zero leakage).
- Every one of the 43 classes appears in *both* the training and validation sets (a naive random shuffle of tracks can accidentally exclude a whole rare class from validation entirely).

This was done by shuffling and splitting tracks *separately within each class*, rather than shuffling all tracks together — a technique called **stratification**.

### 2.5 The retrained model: fixing the frozen-backbone problem

The model was rebuilt with a **two-phase training** approach:

- **Phase 1**: train only the new classification head, backbone still frozen — a fast baseline.
- **Phase 2**: unfreeze the *last four* of the twelve transformer blocks, and continue training the whole thing at a much lower learning rate (to avoid destroying the useful knowledge already in those layers).

**Result: 80.32% test accuracy — and, crucially, validation accuracy (79.4%) and test accuracy (80.3%) were now close together.** That agreement is the real headline of this run: it's direct proof the leakage problem was fixed. The evaluation methodology itself was now trustworthy, even before chasing a higher number.

### 2.6 Reading the training curve like a diagnostic tool

Looking closely at the epoch-by-epoch numbers revealed something important: **validation accuracy was still climbing right up to the last epoch** — the model hadn't finished learning, training had simply hit its pre-set epoch limit. This is a distinct, separate problem from the earlier ones, and a common one: stopping too early leaves real performance on the table.

The fix was to **resume training from the saved checkpoint** for another block of epochs, rather than treating the first result as final.

**Result: 91.08% test accuracy.** The gains only stopped once the validation accuracy genuinely plateaued (and even dipped slightly on the very last epoch) — a real convergence signal, not an arbitrary cutoff.

### 2.7 Reading the mistakes, not just the score

A single accuracy number hides a lot. Digging into exactly *which* images were misclassified (using the confusion matrix and the per-class precision/recall/F1 table) revealed something reassuring: the remaining errors weren't scattered randomly across all 43 classes — they were **concentrated almost entirely in one specific, well-known category**: pairs of signs that are near mirror-images of each other (e.g. "turn left ahead" vs. "turn right ahead," "dangerous curve left" vs. "dangerous curve right"), plus some confusion between similar-looking speed-limit digits.

This matters for two reasons:
1. It confirms the pipeline itself is sound — there's no leftover bug, mislabelling, or leakage causing the errors; the mistakes are exactly the kind of genuinely hard cases the wider research literature on this dataset already documents.
2. It turns a bare percentage into a real finding: *why* the model gets things wrong is often more valuable, academically, than the number itself.

### 2.8 Miscellaneous debugging along the way

A few smaller, instructive bugs came up during implementation:

- **A built-in noise-augmentation layer silently assumed a different data scale than the project was using**, and threw a hard validation error rather than a warning once pushed outside its expected range — a good reminder that library defaults often carry hidden assumptions about your data's format.
- **Locating the model's internal transformer-block layers required reading the actual library source code**, because the model was built using an architecture pattern (Keras's "Functional API") where internal components aren't accessible as simple named attributes the way you might expect — a reminder that when documentation or intuition doesn't match reality, looking directly at the source is often faster than guessing.

---

## Part 3 — Key Takeaways

1. **A validation score that looks *too* good relative to your test score is a warning sign, not a reason to celebrate.** A large gap between them almost always means information is leaking from one side of your split into the other.
2. **"It's training" is not the same as "it's specialising."** A frozen pretrained model can produce a plausible-looking result while still fundamentally not having adapted to your actual task.
3. **Always check whether training stopped because it converged, or because you told it to stop.** These look identical in a single accuracy number, but only one of them means you've found the model's real ceiling.
4. **A confusion matrix tells you more than an accuracy score ever will.** Whether a model's mistakes are random noise or a consistent, explainable pattern changes the whole meaning of the result.
5. **Infrastructure problems (wrong GPU, lost data, slow downloads, broken credentials) will eat as much time as modelling problems, if not more — plan for that.**
6. **When a dataset is rehosted by someone else, verify nothing important was silently dropped** (filenames, structure, metadata) before trusting it.

---

## Glossary

| Term | Plain-English meaning |
|---|---|
| **Accuracy** | % of predictions that were correct |
| **Attention** | The mechanism letting a transformer weigh how relevant different parts of the input are to each other |
| **Backbone** | The main body of a pretrained model, before any task-specific final layer is added |
| **Batch size** | How many images are processed together in one training step |
| **Checkpoint** | A saved snapshot of a model's learned numbers at a point in training |
| **Compute capability** | A number describing which generation of NVIDIA GPU architecture a chip uses — determines what software can run on it |
| **Confusion matrix** | A grid showing what the model predicted vs. what was actually true, for every class |
| **Data leakage** | When information from your test/validation set accidentally influences training, making results look better than they really are |
| **Early stopping** | Automatically halting training once the validation score stops improving, to avoid wasting time or overfitting |
| **Epoch** | One full pass through the entire training dataset |
| **F1-score** | A single number balancing precision and recall |
| **Fine-tuning** | Continuing to train a pretrained model's weights on new, specific data |
| **Frozen (layer)** | A part of the model that is not updated during training |
| **GPU / VRAM** | The specialised chip (and its dedicated memory) used to do the heavy math of training quickly |
| **Learning rate** | How large a step the model takes when updating its numbers after each mistake — too high and it overshoots, too low and it barely learns |
| **Overfitting** | When a model performs well on training data but poorly on new data, because it memorised specifics rather than learning general patterns |
| **Parameters** | The adjustable internal numbers of a model that get tuned during training |
| **Precision** | Of everything the model labelled as class X, what fraction actually was class X |
| **Pretrained model** | A model already trained on a large, general dataset before being adapted to your specific task |
| **Recall** | Of everything that truly was class X, what fraction did the model correctly catch |
| **Stratified split** | Splitting data so that each category is proportionally represented in every resulting subset |
| **Track (in GTSRB)** | A sequence of near-identical video frames of the same physical sign |
| **Transfer learning** | Reusing a model's existing knowledge on a new, related task instead of training from scratch |
| **Transformer block** | One repeating unit of a transformer model's structure, combining attention with further processing |
| **Validation set** | Data held back from training, used to check progress during training (not the final exam) |
| **ViT (Vision Transformer)** | A transformer-based model architecture applied to images instead of text |

---

## Where this could go next

If there's time and appetite for one more experiment, two directions would extend this work meaningfully rather than just chasing a slightly higher number:

- **Unfreeze more of the backbone** (e.g. 8 blocks instead of 4) — most likely to specifically help the remaining mirror-symmetric sign confusions, since that distinction depends on finer spatial detail than the currently-frozen early layers encode.
- **Compare a smaller model (ViT-Small) against the current ViT-Base** — on a benchmark this close to saturation, showing that a much smaller, cheaper model achieves comparable accuracy is a genuinely interesting finding in its own right, and says something real about how much model capacity this particular task actually needs.
