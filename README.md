<div align="center">

# TOP Analysis — Electron + Jets Channel

**Coffea-based framework for charge asymmetry extraction in tt̄ production**

![Python](https://img.shields.io/badge/Python-3.8-blue?style=flat-square&logo=python&logoColor=white)
![Coffea](https://img.shields.io/badge/Coffea-latest-teal?style=flat-square)
![CMS](https://img.shields.io/badge/CMS-Run%202%20UL-blue?style=flat-square)
![NanoAOD](https://img.shields.io/badge/Format-NanoAOD-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-orange?style=flat-square)

</div>

---

## Overview

This repository contains a **Coffea-based analysis framework** for the **Electron + Jets channel** in top quark pair production. The framework extracts **flavour-separated, collider-independent charge asymmetries** (Aᵤ and A_d) in tt̄ production using CMS Run 2 Ultra-Legacy data.

---

## ✨ What This Framework Does

| Module | Description |
|--------|-------------|
| 🔍 **Event & Object Selection** | Trigger requirements, lepton/jet/b-tag selection, scale factors, and pileup weights |
| 📊 **Data / MC Comparison** | Stacked histogram plots with ratio panels, era-parameterized luminosity weighting |
| 🧮 **ABCD QCD Estimation** | Data-driven QCD background estimation using the ABCD transfer factor method |
| 🤖 **BDT Classification** | XGBoost-based binary classifier for qq̄ / gg initial-state enrichment |
| 🧩 **b-tagging Efficiency Maps** | JSON correction maps for b-tagging scale factors via `correctionlib` |
| 📐 **χ² Kinematic Fitting** | Soft-constraint kinematic fitter with L-BFGS-B for top quark mass reconstruction |
| 📈 **TUnfold Unfolding** | Response matrix construction and unfolding pipeline for parton-level asymmetries |

---

## 🔬 Analysis Pipeline

```
NanoAOD Input
     │
     ▼
Event & Object Selection (SFs, triggers, weights)
     │
     ▼
ABCD QCD Background Estimation
     │
     ▼
BDT Training (qq̄ / gg enrichment)
     │
     ▼
χ² Kinematic Fitting (top quark reconstruction)
     │
     ▼
TUnfold Unfolding (response matrix, L-curve scan)
     │
     ▼
Aᵤ and A_d — Parton-level Charge Asymmetries
```

---

## 📦 Dependencies

The analysis requires the following Python packages:

| Package | Purpose |
|---------|---------|
| [coffea](https://coffea-hep.readthedocs.io/en/latest/) | columnar HEP analysis framework |
| [awkward-array](https://pypi.org/project/awkward/) | jagged array operations |
| [dask-awkward](https://docs.dask.org/en/latest/) | lazy/distributed execution |
| [hist](https://hist.readthedocs.io/en/latest/) | histogram objects |
| [correctionlib](https://cms-nanoaod.github.io/correctionlib/install.html) | scale factor corrections (b-tagging, PU, etc.) |
| [numpy](https://numpy.org/) | numerical operations |
| [pyarrow](https://pypi.org/project/pyarrow/) | parquet I/O for BDT inputs |
| [xgboost](https://xgboost.readthedocs.io/) | gradient boosted decision trees |
| [uproot](https://uproot.readthedocs.io/) | ROOT I/O in Python |

---

## 🚀 Installation

It is recommended to use a virtual environment (e.g., Conda).

### Using Conda

```bash
# Create the environment
conda create -n <env_name> python=3.8 \
  conda-forge::uproot \
  conda-forge::root \
  conda-forge::awkward \
  conda-forge::coffea \
  conda-forge::pyarrow

# Activate
conda activate <env_name>

# Install remaining dependencies
pip install correctionlib hist xgboost
```

---

## 📁 Repository Structure

```
top-analysis/
├── processor/          # Coffea processors (selection, weights, ABCD)
├── bdt/                # XGBoost training and evaluation scripts
├── btag_maps/          # b-tagging efficiency JSON maps
├── kinematic_fit/      # χ² kinematic fitter
├── unfolding/          # TUnfold pipeline scripts
├── plots/              # Data/MC and BDT performance plotting scripts
└── README.md
```

---

## 📋 Dataset

- **Format**: NanoAOD (CMS Run 2 Ultra-Legacy)
- **Channel**: Electron + Jets
- **Eras**: 2016 (preVFP + postVFP), 2017, 2018
- **Collaboration**: CMS

---

<div align="center">

**CMS Collaboration · Run 2 UL · Electron + Jets · tt̄ Charge Asymmetry**

</div>

