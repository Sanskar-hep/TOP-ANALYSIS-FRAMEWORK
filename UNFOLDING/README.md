# Unfolding Pipeline

Scripts and workflow for estimating the **truth distribution** from the reconstructed distribution in the electron+jets channel using the **TUnfold** framework.

## Structure

```text
UNFOLDING/
├── response_matrix
│   ├── config_unfolding.yaml
│   └── response_matrix.py
├── TUNFOLD_scripts
│   └── tunfold.py
└── unroll_signal_and_background_histograms
    ├── config_unfolding.yaml
    └── unrolling_final_june_ver2.py
```

## Workflow

### 1. Unroll signal and background histograms

```bash
cd unroll_signal_and_background_histograms
python3 unrolling_final_june_ver2.py
```

Generates the unrolled `N+` and `N-` histograms in bins of `m_ttbar`, saved to a ROOT file.

### 2. Build response matrices

```bash
cd response_matrix
python3 response_matrix.py
```

Generates the nominal and systematic response matrices for the signal and background histograms.

### 3. Collect outputs

```bash
mkdir tunfold_pipeline
```

Copy the ROOT files produced in steps 1 and 2 into `tunfold_pipeline/`.

### 4. Run TUnfold

```bash
cd TUNFOLD_scripts
python3 tunfold.py
```

Produces the unfolded distributions.

## Notes

- Each stage reads its own `config_unfolding.yaml`; keep these in sync when changing binning or input paths.
- Steps 1 and 2 are independent and can be run in either order, but both must complete before step 3.
