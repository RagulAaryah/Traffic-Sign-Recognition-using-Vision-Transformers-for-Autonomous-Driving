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
- **Superseded - not cited by the dissertation.** The 100-epoch
  configuration (Eras 2-4 below) replaced this training regime entirely.
  The chi-squared = 63.3 / p = 1.78x10^-15 figures above have not been
  traced to a specific executed notebook cell and should be independently
  re-verified against raw per-image prediction CSVs before being cited
  anywhere, consistent with the verification standard applied to every
  other figure in this project.

---

## Era 2 - 100-epoch config, first successful run (verified), patience=8

**Commit:** [`65e0677`](https://github.com/RagulAaryah/Traffic-Sign-Recognition-using-Vision-Transformers-for-Autonomous-Driving/commit/65e0677fd6151ed3c6eaa98a2c4e3618f32e2225) - "Guard contrast-correction visualization cell against zero target examples"

This commit's message describes a plotting-bug fix, but the notebook *state*
at this commit is the one that produced this era's result - recorded here
explicitly since the message alone doesn't say so.

- **Test accuracy: 98.88%**
- **Mirror-pair errors: 0**
- Full classification report and confusion matrix generated
- Three-way ablation comparison (baseline / ablation / flip-loss) run and
  recorded against this exact checkpoint in the same session:
  - baseline 98.88% (0 mirror errors) / ablation (lambda=0) 98.50%
    (30 mirror errors) / flip-loss (lambda=0.1) 98.84% (0 mirror errors)
  - McNemar: baseline->ablation chi2=16.73, p=4.3e-05; ablation->flip
    chi2=12.51, p=0.000405
- **Checkpoint independently verified**, not just claimed from notebook
  output:
  - Size: **1,030,589,838 bytes** (982 MB)
  - Modified: **15 August 2026, 20:25:08** (8:25:08 PM)
  - Verified via local file properties, matching the training log's save
    timestamp exactly
  - Local backup filename: `gtsrb_vit_final_98.88_VERIFIED.keras` (not
    committed to this repo - see Open Items)
- **Superseded by Era 4 for dissertation purposes** - patience=15
  (Era 4) is the current training configuration; this era's early-stopping
  patience (8) predates the supervisor-directed increase to 15.

---

## Era 3 - 100-epoch config, second run (after checkpoint overwrite), patience=8

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
- Documented for completeness and as evidence of realistic run-to-run
  variance under the patience=8 configuration; not cited by the
  dissertation.

---

## Era 4 - 100-epoch config, patience raised 8 -> 15 (current, supervisor-directed)

**Commit:** not yet committed at time of writing. Add its hash here once
pushed - see the combined commit covering the stale-baseline-snapshot fix,
the checkpoint-reload fix, the patience change, and the full re-run.

Early-stopping patience raised from 8 to 15 across baseline, ablation, and
flip-loss training, per supervisor direction. Also incorporates two bug
fixes discovered and corrected in the same session:
  - `test_predictions_baseline.csv` was previously only refreshed if
    missing, allowing a stale file from an earlier config to silently
    survive comparison. Now force-refreshed every run.
  - The repaired flip-loss model was at one point evaluated via
    whatever the global `model` variable happened to reference,
    producing a `test_predictions_after_flip.csv` byte-identical to
    the unrepaired baseline. Fixed by loading repaired weights into an
    explicitly named variable with a sanity-check assertion
    (>97.5% accuracy on a 2,000-image subsample) before saving results.

- **Baseline: 98.78% accuracy, 154 total errors, 10 mirror-pair errors**
  (5 in 36/37, 5 in 38/39; 19/20 and 33/34 clean)
- **Ablation (lambda=0): 99.12% accuracy, 111 total errors, 10 mirror-pair
  errors** (redistributed: 3 in 36/37, 7 in 38/39) - does **not** regress
  relative to baseline under this configuration, in contrast to Era 2/3's
  patience=8 runs
- **Flip-loss (lambda=0.1): 98.66% accuracy, 169 total errors, 0
  mirror-pair errors** - the only configuration reaching zero across all
  four pairs
- McNemar: baseline->ablation chi2=21.25, p=4.0e-06 (favours ablation);
  baseline->flip chi2=1.52, p=0.218 (n.s.); ablation->flip chi2=25.79,
  p=3.8e-07 (favours ablation)
- Position-embedding probe re-run on the repaired model: partner-class
  movement increases after repair rather than decreasing (e.g. 38/39
  reverse direction 3.3% -> 80.0%), consistent with the auxiliary loss
  having been learned as intended rather than the repair failing
- Resolution analysis and contrast-correction rescue test regenerated
  against this checkpoint (both were stale, reflecting Era 2's 98.88%
  checkpoint): contrast correction now fixes 4 of 5 target images with
  0 of 90 controls broken (previously reported as 0/15 corrected, an
  artefact of a preprocessing bug since fixed)
- **This is the primary result set referenced in the dissertation.**
  The central finding under this configuration is a trade-off rather
  than a protective effect: flip-loss removes all mirror-pair errors at
  a 0.46-point accuracy cost relative to the ablation, rather than
  preventing a regression the ablation would otherwise show (contrast
  with Era 2/3, patience=8, where continued training without the
  auxiliary term did regress).

---

## Which result to cite

- **Dissertation, primary and only citation target: Era 4**, commit
  TBC (patience=15: baseline 98.78% / ablation 99.12% / flip-loss 98.66%,
  10 -> 0 mirror-pair errors under flip-loss, trade-off framing per
  Chapter 6)
- **Era 1** is superseded. Its McNemar figures (chi2=63.3) are unverified
  against raw per-image outputs and should not be cited without that
  verification.
- **Era 2** is superseded for dissertation purposes but remains the most
  independently verified patience=8 checkpoint (file size/mtime cross-checked),
  useful as a reference point if a patience=8 vs patience=15 comparison is
  ever written up.
- **Era 3** is documented for completeness and as evidence of run-to-run
  variance under patience=8; not recommended as a citation target under any
  circumstance.

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
- The Era 4 commit hash should be added to this file once the combined
  commit (bug fixes + patience change + re-run) is pushed
- Era 1's chi2=63.3 McNemar figure should be re-verified against raw
  per-image prediction CSVs, or removed from this file, before any future
  reader treats it as an established result
