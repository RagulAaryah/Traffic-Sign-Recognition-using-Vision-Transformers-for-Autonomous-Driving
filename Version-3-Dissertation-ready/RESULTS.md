# Results Summary

This file is the canonical record of model results across this project's
development history. All notebook changes are committed sequentially to the
single file `gtsrb_vit_FINAL.ipynb`, so individual results are identified here
by **commit hash**, not filename. Commit messages describe *code* changes;
this file describes *what those changes actually produced*.

Full history: https://github.com/RagulAaryah/Traffic-Sign-Recognition-using-Vision-Transformers-for-Autonomous-Driving/commits/main/Version-3-Dissertation-ready/gtsrb_vit_FINAL.ipynb

---

## Era 1 - Original config (15/6 epochs, per-stage patience)

**Commit:** [`821a358`](https://github.com/RagulAaryah/Traffic-Sign-Recognition-using-Vision-Transformers-for-Autonomous-Driving/commit/821a358ac440229f53d7144c53e3aaa74979a802) - "Baseline: 15/6-epoch training config with per-stage patience (3/4/3), pre-supervisor-directed epoch increase"

- **Test accuracy: 97.12%**
- **Mirror-pair errors: 102 / 364 total errors (28.0%)**
- After the flip-contrastive repair: mirror-pair errors reduced to **2**
  (-98.0%), overall accuracy **98.67%**
- Ablation (lambda=0) control: mirror-pair errors only reduced to 70,
  confirming the repair's effect is orientation-specific (McNemar
  chi-squared = 63.3, p = 1.78 x 10^-15 on mirror-pair classes)
- **This is the primary result set referenced in the dissertation.**

---

## Era 2 - 100-epoch config, first successful run (verified)

**Commit:** [`65e0677`](https://github.com/RagulAaryah/Traffic-Sign-Recognition-using-Vision-Transformers-for-Autonomous-Driving/commit/65e0677fd6151ed3c6eaa98a2c4e3618f32e2225) - "Guard contrast-correction visualization cell against zero target examples"

This commit's message describes a plotting-bug fix, but the notebook *state*
at this commit is the one that produced the project's strongest verified
result - recorded here explicitly since the message alone doesn't say so.

- **Test accuracy: 98.88%**
- **Mirror-pair errors: 0**
- Full classification report and confusion matrix generated
- Three-way ablation comparison (baseline / ablation / flip-loss) run and
  recorded against this exact checkpoint in the same session
- **Checkpoint independently verified**, not just claimed from notebook
  output:
  - Size: **1,030,589,838 bytes** (982 MB)
  - Modified: **15 August 2026, 20:25:08** (8:25:08 PM)
  - Verified via local file properties, matching the training log's save
    timestamp exactly
  - Local backup filename: `gtsrb_vit_final_98.88_VERIFIED.keras` (not
    committed to this repo - see Open Items)

---

## Era 3 - 100-epoch config, second run (after checkpoint overwrite)

**Commit:** not yet present in the pushed history at time of writing (the
mirror-pair-tracker / directional-regression-analysis work, run after commit
`65e0677`). Add its hash here once committed.

- **Test accuracy: 98.41%**
- Best epoch: 33 (val_accuracy 0.99591), restored via EarlyStopping
- **Notable confusions (not present in Era 2):**
  - `class 36 -> class 37`: 22 (unidirectional - matches the original
    directional pattern)
  - `class 11 -> class 37`: 9 (a new confusion, unrelated to the mirror-pair
    classes)
- Produced as a side effect of adding a per-epoch `MirrorPairTracker`
  callback to Phase 2 and retraining to capture the mirror-pair error curve
  across training - retraining was required because this data cannot be
  recovered retroactively from a completed run
- **Cause of the overwrite:** the checkpoint save path (`gtsrb_vit_final.keras`
  on the training pod) is not run-tagged, so re-running Phase 2 for any reason
  silently replaces the previous checkpoint. This is a known gap - see Open
  Items below. Note this is a pod-storage issue, unrelated to git history.

---

## Which result to cite

- **Dissertation, primary claims:** Era 1, commit `821a358` (97.12% -> 98.67%
  after repair, 102 -> 2 mirror-pair errors, ablation-confirmed causal effect)
- **If citing the 100-epoch exploration:** Era 2, commit `65e0677` (98.88%,
  0 mirror-pair errors) - this is the era with a verified, independently
  checked checkpoint and a complete evaluation trail
- **Era 3** is documented for completeness and as evidence of realistic
  run-to-run variance under this training configuration; not recommended as
  a primary citation, since its checkpoint further overwrote Era 2's on the
  training pod, and Era 2 remains the more complete, independently verified
  result

---

## Open items / lessons for future runs

- Checkpoint filenames on the training pod should be timestamped or
  run-tagged (e.g. `gtsrb_vit_final_{timestamp}.keras`) to prevent silent
  overwrites - this caused real confusion across Eras 2 and 3 above, and is
  a pod-storage issue separate from git history
- The mirror-pair error curve (Era 3) showed noisy, non-monotonic decline
  rather than a clean convergence point - worth treating with the same
  run-to-run-variance caveat already documented for other results in this
  project
- The Era 3 commit hash should be added to this file once located/confirmed
  in the repo's history
