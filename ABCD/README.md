# ABCD Method

This directory contains the scripts required to estimate the QCD multijet background using the ABCD method.

## Directory Structure

```text
ABCD/
├── Data_MC_agreement/
│   ├── convert_qcd.py
│   ├── convert_to_root.py
│   ├── plot_stacked_histograms_thesis.C
│   └── run_plots_THStack.sh
├── pre-requisites/
│   ├── ABCD_runner.sh
│   ├── region_abcd_proc.py
│   └── region_runner.py
├── QCD_estimation/
│   ├── difference_hist_new.py
│   └── transfer_fac_bb.py
└── README.md
```

## Workflow

### Step 1: Prepare the ABCD inputs

Navigate to the `pre-requisites` directory and execute:

```bash
cd pre-requisites
./ABCD_runner.sh
```

This script processes the Coffea outputs and prepares all the inputs required for the ABCD background estimation.

---

### Step 2: Estimate the QCD background

Move to the `QCD_estimation` directory:

```bash
cd ../QCD_estimation
```

Compute the difference between data and the non-QCD Monte Carlo prediction for the control regions **A** and **C**:

```bash
python3 difference_hist_new.py A
python3 difference_hist_new.py C
```

These commands produce two JSON files containing the background-subtracted distributions for Regions **A** and **C**.

Next, calculate the transfer factors:

```bash
python3 transfer_fac_bb.py regionC regionA
```

This script uses the Region C and Region A distributions to compute the QCD transfer factors and produces the corresponding JSON file.

---

### Step 3: Produce the final Data–MC comparison plots

Navigate to the `Data_MC_agreement` directory:

```bash
cd ../Data_MC_agreement
```

Convert the Coffea outputs to ROOT format:

```bash
python3 convert_qcd.py
python3 convert_to_root.py
```

Copy the transfer-factor JSON file generated in the previous step into the `Data_MC_agreement` directory.

Finally, produce the stacked Data–MC comparison plots:

```bash
./run_plots_THStack.sh <ERA_NAME>
```

where `<ERA_NAME>` is the data-taking era (for example, `2016preVFP`, `2016postVFP`, `2017`, or `2018`).
