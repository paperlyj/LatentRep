## Download the complete materials

The complete manuscript submission snapshot is available as
`LatentRep_CILS_Reproducibility.zip` under **Releases → v1.0.0**.

Download and extract this archive before running the verification scripts.
The repository-root files are provided for browsing and do not include
the complete saved results.

See `EVIDENCE_STATUS.md` for the scope and limitations of reproducibility.

# LatentRep analysis and reproducibility materials

Companion to the manuscript submitted for consideration to Chemometrics and Intelligent Laboratory Systems.

## Read first
This archive contains an unchanged historical analysis notebook and executable, pinned post hoc sensitivity scripts with saved outputs. It does NOT establish exact recovery of the original 68-group prostate partition. See EVIDENCE_STATUS.md for the boundary between retained historical evidence and independently checkable post hoc results.

## Offline verification of Table 3
Use Python 3.12.14 and install requirements-sensitivity.txt, then run:

    python verify_stored_outputs.py

This verifies stored F1, label hashes and fold group isolation for 220 prediction vectors. Results for seeds 0–9 determine Table 3; seed 2026 is checked separately and excluded from that average. evaluation_labels.npz contains the targets and recorded identifiers for this check, without spectral intensities. The executed check is saved in evidence/stored_output_verification.json.

## Retrieve inputs and rerun the post hoc analyses

    python -m pip install -r requirements-sensitivity.txt
    python prepare_inputs.py
    python verify_table3.py

The public source archive is SHA-256 checked; the 63 Figshare prostate files are MD5 checked. Raw spectra are retrieved from source repositories and not redistributed. Input matrix hashes are recorded under results/ and must match before comparing output.

On Linux (multiprocessing fork):

    export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
    for task in covid19_salvia parkinson serum_prostate_cancer alzheimer
    do
        python matched_sensitivity.py inputs/$task.npz rerun/$task --workers 3
    done

Each task contains the two 51-resolution paths, selected original-rule/index-rule/k-means groups, fold assignments, predictions and F1 by seed. Original-rule discards the first returned neighbor; index-rule excludes the query by index. Group labels are converted to strings before StratifiedGroupKFold. All errors are paired with the same-environment Oracle. These are fixed-logistic-regression sensitivity analyses, separate from the original seven-model benchmark.

## Historical notebook and figures
LatentRep_benchmark.ipynb is preserved as received. Running its cells requires the data loader and original workflow dependencies; the complete original environment was not recorded. The frozen prostate control depends on the live original benchmark object and may not yield historical values after rerunning inference. The archive preserves the original results rather than substituting fresh partitions.

figures/ contains the manuscript images and the original/retained Fig. 5 sources. negative_control_printed_summary.csv contains rounded summary statistics only, not raw permutation data. Read AI_AND_FIGURE_PROVENANCE.md for AI use and figure provenance. SHA256SUMS.txt covers all supplied files except itself.
