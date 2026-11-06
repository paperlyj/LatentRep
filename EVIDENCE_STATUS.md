# Evidence status — 6 November 2026

## Historical benchmark
The supplied notebook is unchanged and is byte-identical to the author's LatentRep(5).ipynb. Its source and retained text outputs are extracted under evidence/. Cell numbers are zero-based notebook cell indices, not execution counts. These are retained historical outputs, not newly rerun outputs.

- Cell 19 (Step 18): original external benchmark, including prostate resolution 6.00 and 68 groups, ARI 0.6480 and seven-model error summaries.
- Cell 38 (Step 26): a separate inference call, including the discrepant prostate selection at 5.80 with 68 groups.
- Cell 39 (Step 26B): retained confirmation that the original in-memory Step 18 prostate groups were reused; fixed logistic-regression error 0.0394, Random error 0.2410, both empirical p = 0.010.
- Cell 40: original Fig. 5 plotting code and retained raster image. The boxplot uses whis=(2.5, 97.5), medians and interquartile boxes, with 99 individual permutation points per control.
- Cells 25 and subsequent simulation-analysis cells: retained summaries of the 405 scenarios, low-ARI subset and persistence analyses.

NOT recovered: a serialized row-aligned historical prostate 68-group vector, the complete original dependency/environment lock, historical per-fold predictions for the seven-model benchmark, and the complete historical 99-value permutation vectors needed to reconstruct every Fig. 5 box and point. Notebook code and printed summaries support reported results but do not replace these missing numerical objects. The full historical pipeline has not been independently reproduced in this package.

## Fresh matched analysis
results/ contains independently calculated post hoc partitions, five-fold assignments, predictions and F1 for split seeds 0–9 and 2026. The historical partition is not among these group vectors. Fresh prostate has 69 groups at resolution 5.80 for row-major input. Explicit index-based self-exclusion removes 100 self-loops while leaving the selected memberships unchanged (ARI = 1). This demonstrates the outcome of that matched comparison; it does not explain or recover the historical 68 groups.

The supplied offline verifier recalculates all 220 stored prediction-vector F1 scores, checks group isolation, verifies target/group hashes and compares Table 3 rounding. It does not refit models or rerun Louvain. evidence/input_hash_verification.json records a separate check of X, y and g against the existing reconstructed public inputs.

## Figures and summary data
Figure_5_as_submitted.png is copied byte-for-byte from the current author manuscript. Figure_5_author_original.png is the user's larger original image. Figure_5_retained_notebook.png is extracted from cell 40. None was reconstructed from rounded statistics. negative_control_printed_summary.csv contains rounded means and percentiles only; it cannot reconstruct medians, quartiles, whisker endpoints or individual permutations. The earlier mean/interval redrawing is not part of the submitted main manuscript.

## Before claiming exact historical reproducibility
Recover the original group vector together with its row-order mapping, matrix hash, selected resolution and environment. Recover the original permutation arrays and export Fig. 5 directly at publication resolution. If these objects cannot be recovered, a new locked complete benchmark must be reported as a replacement analysis, with all dependent text/tables/figures updated together. Do not relabel fresh 69-group outputs as historical 68-group evidence.
