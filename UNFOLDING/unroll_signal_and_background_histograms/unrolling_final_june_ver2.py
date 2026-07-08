import os
import h5py
import yaml
import numpy as np
import ROOT

ROOT.gROOT.SetBatch(True)

# ─────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────
def load_config(config_path, era):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    def str_keys(d):
        return {str(k): v for k, v in d.items()}

    cfg["eras"] = str_keys(cfg["eras"])
    for ds in cfg["datasets"].values():
        ds["xsec"] = str_keys(ds["xsec"])
        ds["ngen"]  = str_keys(ds["ngen"])

    era_cfg     = cfg["eras"][era]
    Y_0         = cfg["physics"]["Y_0"]
    pgof_cut    = cfg["physics"]["pgof_cut"]
    LUMI        = era_cfg["lumi"]
    OUTDIR      = era_cfg["outdir"]
    RECO_DIR    = era_cfg["reco_dir"]
    GEN_DIR     = era_cfg["gen_dir"]
    FILE_SUFFIX = era_cfg["file_suffix"]
    GEN_SUFFIX  = era_cfg["gen_suffix"]

    reco_mtt_edges = np.array(cfg["binning"]["reco"]["mtt_edges"], dtype=float)
    gen_mtt_edges  = np.array(cfg["binning"]["gen"]["mtt_edges"],  dtype=float)

    datasets = {}
    for name, ds in cfg["datasets"].items():
        datasets[name] = {
            "xsec"      : ds["xsec"][era],
            "ngen"      : ds["ngen"][era],
            "is_signal" : ds["is_signal"],
        }

    return (Y_0, pgof_cut, LUMI, OUTDIR, RECO_DIR, GEN_DIR,
            FILE_SUFFIX, GEN_SUFFIX,
            reco_mtt_edges, gen_mtt_edges, datasets)


# ─────────────────────────────────────────────
# ERA selection — change only this line
# ─────────────────────────────────────────────
ERA         = "2018"
CONFIG_PATH = "config_unfolding.yaml"

(Y_0, pgof_cut, LUMI, OUTDIR, RECO_DIR, GEN_DIR,
 FILE_SUFFIX, GEN_SUFFIX,
 reco_mtt_edges, gen_mtt_edges, UL_datasets) = load_config(CONFIG_PATH, ERA)

os.makedirs(OUTDIR, exist_ok=True)

# ─────────────────────────────────────────────
# Systematic sources — (up, down) weight-field pairs
# ─────────────────────────────────────────────
SYST_PAIRS = [
    ("pileupWeight_up",      "pileupWeight_down"),
    ("PreFiringWeight_up",   "PreFiringWeight_down"),
    ("eleID_up",             "eleID_down"),
    ("btag_sf_up",           "btag_sf_down"),
    ("pileUp_sf_up",         "pileUp_sf_down"),
]

# ─────────────────────────────────────────────
# Kinematics helpers
# ─────────────────────────────────────────────
def inv_mass(px, py, pz, E):
    return np.sqrt(np.maximum(E**2 - (px**2 + py**2 + pz**2), 0.0))

def rapidity(E, pz):
    return 0.5 * np.log((E + pz) / (E - pz))

def assign_top_antitop(cond, lep, had):
    return np.where(cond, lep, had), np.where(cond, had, lep)

def make_Nplus_Nminus(y_top, y_antitop, Y_0, label=""):
    top_fwd     = np.abs(y_top)     > Y_0
    antitop_fwd = np.abs(y_antitop) > Y_0
    Nplus  = top_fwd  & ~antitop_fwd
    Nminus = antitop_fwd & ~top_fwd
    print(f"  [{label}] N+={np.sum(Nplus)}  N-={np.sum(Nminus)}")
    return Nplus, Nminus

# ─────────────────────────────────────────────
# Histogram-filling helpers
# ─────────────────────────────────────────────
def make_labels(mtt_edges):
    n, lbl = len(mtt_edges) - 1, []
    for sign in ["N_{+}", "N_{-}"]:
        for i in range(n):
            lbl.append(f"{sign}({int(mtt_edges[i])}-{int(mtt_edges[i+1])})")
    return lbl

def fill_unrolled(mtt, Nplus, Nminus, mtt_edges, w):
    """Weighted counts per unrolled bin (N+ block, then N- block)."""
    n_mtt  = len(mtt_edges) - 1
    counts = np.zeros(2 * n_mtt, dtype=float)
    for i in range(n_mtt):
        lo, hi          = mtt_edges[i], mtt_edges[i + 1]
        in_mtt          = (mtt >= lo) & (mtt < hi)
        counts[i]       = np.sum(w[in_mtt & Nplus])
        counts[n_mtt+i] = np.sum(w[in_mtt & Nminus])
    return counts

def fill_stat_errors(mtt, Nplus, Nminus, mtt_edges, w):
    """MC statistical uncertainty per bin: sqrt(sum w^2)."""
    n_mtt  = len(mtt_edges) - 1
    errors = np.zeros(2 * n_mtt, dtype=float)
    for i in range(n_mtt):
        lo, hi          = mtt_edges[i], mtt_edges[i + 1]
        in_mtt          = (mtt >= lo) & (mtt < hi)
        errors[i]       = np.sqrt(np.sum(w[in_mtt & Nplus]  ** 2))
        errors[n_mtt+i] = np.sqrt(np.sum(w[in_mtt & Nminus] ** 2))
    return errors

def fill_th1(name, title, counts, errors, labels):
    """Build a TH1D with given bin contents/errors and axis labels."""
    n = len(counts)
    h = ROOT.TH1D(name, title, n, 0, n)
    h.Sumw2()
    for i, (c, e) in enumerate(zip(counts, errors)):
        h.SetBinContent(i + 1, c)
        h.SetBinError(i + 1, e)
    for i, lbl in enumerate(labels):
        h.GetXaxis().SetBinLabel(i + 1, lbl)
    return h

# ─────────────────────────────────────────────
# Write the nominal + systematic histograms for one
# dataset, at one level ("reco" or "gen"):
#
#   <dataset>_<level>_nominal      content + MC stat error
#   <dataset>_<level>_<source>Up   Up counts
#   <dataset>_<level>_<source>Down Down counts
# ─────────────────────────────────────────────
def write_dataset_histograms(fout, dataset, level,
                              mtt, Nplus, Nminus, mtt_edges,
                              w_nominal, weights, lumi_scale,
                              syst_pairs, available_systs, labels):
    fout.cd()

    # (a) Nominal — content + stat error
    counts_nom = fill_unrolled(mtt, Nplus, Nminus, mtt_edges, w_nominal)
    errors_nom = fill_stat_errors(mtt, Nplus, Nminus, mtt_edges, w_nominal)
    h_nom = fill_th1(f"{dataset}_{level}_nominal",
                      f"{dataset} {level} nominal",
                      counts_nom, errors_nom, labels)
    h_nom.Write()

    # (b) Systematics — content only, no stat error (standard convention)
    n_written = 0
    for up_name, dn_name in syst_pairs:
        source = up_name.replace("_up", "")

        if up_name not in available_systs or dn_name not in available_systs:
            print(f"    [SKIP] {source}: not found for {dataset} ({level})")
            continue

        w_up = weights[up_name] * lumi_scale
        w_dn = weights[dn_name] * lumi_scale

        counts_up = fill_unrolled(mtt, Nplus, Nminus, mtt_edges, w_up)
        counts_dn = fill_unrolled(mtt, Nplus, Nminus, mtt_edges, w_dn)

        h_up = fill_th1(f"{dataset}_{level}_{source}Up",
                         f"{dataset} {level} {source} Up",
                         counts_up, np.zeros_like(counts_up), labels)
        h_dn = fill_th1(f"{dataset}_{level}_{source}Down",
                         f"{dataset} {level} {source} Down",
                         counts_dn, np.zeros_like(counts_dn), labels)
        h_up.Write()
        h_dn.Write()
        n_written += 1

    print(f"  Wrote {dataset}_{level}_nominal + "
          f"{n_written} systematic pair(s) (Up/Down)")


# ─────────────────────────────────────────────
# Output ROOT file
# ─────────────────────────────────────────────
outpath = os.path.join(OUTDIR, "unrolled_histograms_nom_and_sys.root")
fout = ROOT.TFile(outpath, "RECREATE")
print(f"Output ROOT file: {outpath}")

reco_labels = make_labels(reco_mtt_edges)
gen_labels  = make_labels(gen_mtt_edges)

# ─────────────────────────────────────────────
# Loop over all datasets
# ─────────────────────────────────────────────
for dataset, ds_info in UL_datasets.items():
    xsec      = ds_info["xsec"]
    ngen      = ds_info["ngen"]
    is_signal = ds_info["is_signal"]

    print(f"\n{'='*60}")
    print(f"Processing: {dataset}  (era={ERA})")

    reco_file = f"{RECO_DIR}/{dataset}{FILE_SUFFIX}"
    gen_file  = f"{GEN_DIR}/{dataset}{GEN_SUFFIX}"

    # ── Load reco file ──────────────────────────────────
    with h5py.File(reco_file, "r") as f:
        best_perm  = f["best_perm"][:]
        best_chi2  = f["best_chi2"][:]
        top_lep_px = f["top_lep_px"][:]
        top_lep_py = f["top_lep_py"][:]
        top_lep_pz = f["top_lep_pz"][:]
        top_lep_E  = f["top_lep_E"][:]
        top_had_px = f["top_had_px"][:]
        top_had_py = f["top_had_py"][:]
        top_had_pz = f["top_had_pz"][:]
        top_had_E  = f["top_had_E"][:]

    # ── Load gen file ───────────────────────────────────
    with h5py.File(gen_file, "r") as f:
        charge          = f["charge"][:]
        weights_full    = f["weight_variations"][:]
        available_systs = list(weights_full.dtype.names)

        if is_signal:
            gen_top_pt          = f["gen_top_pt"][:]
            gen_top_E_arr       = f["gen_top_E"][:]
            gen_top_cosphi      = f["gen_top_cosphi"][:]
            gen_top_sinphi      = f["gen_top_sinphi"][:]
            gen_top_sinheta     = f["gen_top_sinheta"][:]
            gen_antitop_pt      = f["gen_antitop_pt"][:]
            gen_antitop_E_arr   = f["gen_antitop_E"][:]
            gen_antitop_cosphi  = f["gen_antitop_cosphi"][:]
            gen_antitop_sinphi  = f["gen_antitop_sinphi"][:]
            gen_antitop_sinheta = f["gen_antitop_sinheta"][:]

    lumi_scale = (xsec * LUMI) / ngen

    # ── pgof mask ──────────────────────────────────────
    Nevt        = best_perm.shape[0]
    idx         = np.arange(Nevt)
    charge_best = charge[idx, best_perm]
    pgof        = np.exp(-0.5 * best_chi2)
    mask        = (pgof > pgof_cut) & (best_chi2 > 0)

    weights_masked = weights_full[mask]

    # ── RECO kinematics ────────────────────────────────
    cond = charge_best > 0
    top_px, antitop_px = assign_top_antitop(cond, top_lep_px, top_had_px)
    top_py, antitop_py = assign_top_antitop(cond, top_lep_py, top_had_py)
    top_pz, antitop_pz = assign_top_antitop(cond, top_lep_pz, top_had_pz)
    top_E,  antitop_E  = assign_top_antitop(cond, top_lep_E,  top_had_E)

    mtt_reco       = inv_mass(top_px + antitop_px,
                               top_py + antitop_py,
                               top_pz + antitop_pz,
                               top_E  + antitop_E)[mask]
    y_top_reco     = rapidity(top_E,     top_pz)[mask]
    y_antitop_reco = rapidity(antitop_E, antitop_pz)[mask]

    print("  --- Reco ---")
    Nplus_reco, Nminus_reco = make_Nplus_Nminus(
        y_top_reco, y_antitop_reco, Y_0, label="Reco")

    w_nominal_reco = weights_masked["nominal"] * lumi_scale

    write_dataset_histograms(
        fout, dataset, "reco",
        mtt_reco, Nplus_reco, Nminus_reco, reco_mtt_edges,
        w_nominal_reco, weights_masked, lumi_scale,
        SYST_PAIRS, available_systs, reco_labels
    )

    # ── GEN: signal only ───────────────────────────────
    if is_signal:
        gen_top_px_best     = gen_top_pt[idx, best_perm]     * gen_top_cosphi[idx, best_perm]
        gen_top_py_best     = gen_top_pt[idx, best_perm]     * gen_top_sinphi[idx, best_perm]
        gen_top_pz_best     = gen_top_pt[idx, best_perm]     * gen_top_sinheta[idx, best_perm]
        gen_top_E_best      = gen_top_E_arr[idx, best_perm]

        gen_antitop_px_best = gen_antitop_pt[idx, best_perm] * gen_antitop_cosphi[idx, best_perm]
        gen_antitop_py_best = gen_antitop_pt[idx, best_perm] * gen_antitop_sinphi[idx, best_perm]
        gen_antitop_pz_best = gen_antitop_pt[idx, best_perm] * gen_antitop_sinheta[idx, best_perm]
        gen_antitop_E_best  = gen_antitop_E_arr[idx, best_perm]

        mtt_gen       = inv_mass(gen_top_px_best     + gen_antitop_px_best,
                                  gen_top_py_best     + gen_antitop_py_best,
                                  gen_top_pz_best     + gen_antitop_pz_best,
                                  gen_top_E_best      + gen_antitop_E_best)[mask]
        y_top_gen     = rapidity(gen_top_E_best,     gen_top_pz_best)[mask]
        y_antitop_gen = rapidity(gen_antitop_E_best, gen_antitop_pz_best)[mask]

        print("  --- Gen ---")
        Nplus_gen, Nminus_gen = make_Nplus_Nminus(
            y_top_gen, y_antitop_gen, Y_0, label="Gen")

        w_nominal_gen = weights_masked["nominal"] * lumi_scale

        write_dataset_histograms(
            fout, dataset, "gen",
            mtt_gen, Nplus_gen, Nminus_gen, gen_mtt_edges,
            w_nominal_gen, weights_masked, lumi_scale,
            SYST_PAIRS, available_systs, gen_labels
        )

fout.Close()
print(f"\n{'='*60}")
print(f"All done. ROOT file: {outpath}")

if np.any(best_perm == -1):
    print("Warning: best_perm contains -1 values, indicating no valid permutation for some events.")
