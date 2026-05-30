<div align="center">

# Collider-Independent Asymmetries In Top-Quark Pair Production 


![Python](https://img.shields.io/badge/Python-3.8-blue?style=flat-square&logo=python&logoColor=white)
![Coffea](https://img.shields.io/badge/Coffea-latest-teal?style=flat-square)
![CMS](https://img.shields.io/badge/CMS-Run%202%20UL-blue?style=flat-square)
![NanoAOD](https://img.shields.io/badge/Format-NanoAOD-green?style=flat-square)

</div>

---

## Overview

This repository contains a **Coffea-based analysis framework** for the **Electron + Jets channel** in top quark pair production. The framework extracts **flavour-separated, collider-independent asymmetries observables** (Aᵤ and A_d) in tt̄ production using CMS Run 2 Ultra-Legacy data.

---

## ✨ What This Framework Does

| Module | Description |
|--------|-------------|
| 🔍 **Event & Object Selection** | Selection Cuts on electrons, Jets and MET|
| 📊 **Data / MC Comparison** | Data-MC agreement studies for each era of Run2 |
| 🧮 **ABCD QCD Estimation** | Data-driven QCD background estimation using the ABCD method |
| 🤖 **BDT Classification** | XGBoost-based binary classifier for distinguishing top quark pair arising from qq̄(signal) from those of gg|
| 📐 **χ² Kinematic Fitting** | Reconstruction of the top-quark and the antitop quark kinematic observables|
| 📈 **TUnfold Unfolding** | Response matrix construction and unfolding pipeline for parton-level asymmetries (Work in progress)|

---

## 🔬 Analysis Pipeline

```
NanoAOD Input
     │
     ▼
Event & Object Selection 
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
Aᵤ and A_d — Parton-level Asymmetries
```

---

## 📦 Dependencies

The following pre-requisites must be downloaded before running any scripts:

| Package | 
|---------|
| [coffea](https://coffea-hep.readthedocs.io/en/latest/) |
| [awkward-array](https://pypi.org/project/awkward/) |
| [dask-awkward](https://docs.dask.org/en/latest/) |
| [hist](https://hist.readthedocs.io/en/latest/) |
| [correctionlib](https://cms-nanoaod.github.io/correctionlib/install.html)| 
| [numpy](https://numpy.org/) | 
| [xgboost](https://xgboost.readthedocs.io/) | 
| [uproot](https://uproot.readthedocs.io/) |

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


## 📋 Dataset

- **Format**: NanoAOD (CMS Run 2 Ultra-Legacy)
- **Channel**: Electron + Jets
- **Eras**: 2016 (preVFP + postVFP), 2017, 2018
- **Collaboration**: CMS

---

<div align="center">




