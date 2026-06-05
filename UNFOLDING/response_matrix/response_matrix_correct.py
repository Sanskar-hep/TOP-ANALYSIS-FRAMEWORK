import os
import h5py
import yaml
import numpy as np
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)


def load_config(config_path, era):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    def str_keys(d):
        return {str(k): v for k, v in d.items()}

    cfg["eras"] = str_keys(cfg["eras"])
    for ds in cfg["datasets"].values():
        ds["xsec"] = str_keys(ds["xsec"])
        ds["ngen"] = str_keys(ds["ngen"])

    era_cfg = cfg["eras"][era]
    signal_datasets = {
        name: ds for name, ds in cfg["datasets"].items()
        if ds["is_signal"]
    }
    if len(signal_datasets) != 1:
        raise RuntimeError(
            f"Expected exactly one signal dataset, found {list(signal_datasets)}"
        )

    signal = list(signal_datasets)[0]
    signal_cfg = signal_datasets[signal]

    return {
        "Y_0": cfg["physics"]["Y_0"],
        "pgof_cut": cfg["physics"]["pgof_cut"],
        "lumi": float(era_cfg["lumi"]),
        "reco_dir": era_cfg["reco_dir"],
        "gen_dir": era_cfg["gen_dir"],
        "file_suffix": era_cfg["file_suffix"],
        "gen_suffix": era_cfg["gen_suffix"],
        "reco_mtt_edges": np.array(cfg["binning"]["reco"]["mtt_edges"], dtype=float),
        "gen_mtt_edges": np.array(cfg["binning"]["gen"]["mtt_edges"], dtype=float),
        "signal": signal,
        "xsec": float(signal_cfg["xsec"][era]),
        "ngen": int(signal_cfg["ngen"][era]),
    }


def inv_mass(px, py, pz, E):
    return np.sqrt(np.maximum(E**2 - (px**2 + py**2 + pz**2), 0.0))


def rapidity(E, pz):
    return 0.5 * np.log((E + pz) / (E - pz))


def assign_top_antitop(cond, lep, had):
    return np.where(cond, lep, had), np.where(cond, had, lep)


def make_labels(mtt_edges):
    labels = []
    n_mtt = len(mtt_edges) - 1
    for sign in ["N_{+}", "N_{-}"]:
        for i in range(n_mtt):
            labels.append(f"{sign}({int(mtt_edges[i])}-{int(mtt_edges[i + 1])})")
    return labels


def make_Nplus_Nminus(y_top, y_antitop):
    top_fwd = np.abs(y_top) > Y_0
    antitop_fwd = np.abs(y_antitop) > Y_0
    return top_fwd & ~antitop_fwd, antitop_fwd & ~top_fwd


def get_unrolled_bin(mtt, is_Nplus, is_Nminus, mtt_edges):
    n_mtt = len(mtt_edges) - 1
    result = np.full(len(mtt), -1, dtype=int)

    for i in range(n_mtt):
        lo, hi = mtt_edges[i], mtt_edges[i + 1]
        in_mtt = (mtt >= lo) & (mtt < hi)
        result[in_mtt & is_Nplus] = i
        result[in_mtt & is_Nminus] = n_mtt + i

    return result


def make_response_matrix(name, title, with_stat_errors, n_gen_bins, n_reco_bins, gen_edges, reco_edges):
    h = ROOT.TH2D(name, title, n_gen_bins, 0, n_gen_bins, n_reco_bins, 0, n_reco_bins)
    if with_stat_errors:
        h.Sumw2()

    for i, label in enumerate(make_labels(gen_edges)):
        h.GetXaxis().SetBinLabel(i + 1, label)
    for i, label in enumerate(make_labels(reco_edges)):
        h.GetYaxis().SetBinLabel(i + 1, label)

    h.SetMinimum(0.0)
    h.SetOption("COLZ0")
    return h


def fill_response_matrix(h, gen_bin, reco_bin, weights):
    # Maps valid bins to BinCenter (+0.5) and invalid/failed bins (-1) to Underflow (-0.5)
    g_fill = np.where(gen_bin >= 0, gen_bin + 0.5, -0.5)
    r_fill = np.where(reco_bin >= 0, reco_bin + 0.5, -0.5)
    
    for g, r, w in zip(g_fill, r_fill, weights):
        h.Fill(float(g), float(r), float(w))


def zero_bin_errors(h):
    for gx in range(1, h.GetNbinsX() + 1):
        for ry in range(1, h.GetNbinsY() + 1):
            h.SetBinError(gx, ry, 0.0)


def set_matrix_draw_style(h):
    zmax = h.GetMaximum()
    h.SetMinimum(-0.01 * max(1.0, zmax))
    h.SetMaximum(zmax)
    h.SetOption("COLZ0")


# --- CONFIGURATION ---
ERA = "2018"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config_unfolding.yaml")

cfg = load_config(CONFIG_PATH, ERA)

Y_0 = cfg["Y_0"]
pgof_cut = cfg["pgof_cut"]
LUMI = cfg["lumi"]
SIGNAL = cfg["signal"]
XSEC = cfg["xsec"]
NGEN = cfg["ngen"]

RECO_FILE = os.path.join(cfg["reco_dir"], f"{SIGNAL}{cfg['file_suffix']}")
GEN_FILE = os.path.join(cfg["gen_dir"], f"{SIGNAL}{cfg['gen_suffix']}")
OUTDIR = os.path.join(SCRIPT_DIR, "results_response_matrix", ERA)
OUT_FILE = os.path.join(OUTDIR, f"response_matrix_for_{ERA}.root")
os.makedirs(OUTDIR, exist_ok=True)

reco_mtt_edges = cfg["reco_mtt_edges"]
gen_mtt_edges = cfg["gen_mtt_edges"]
n_reco_bins = 2 * (len(reco_mtt_edges) - 1)
n_gen_bins = 2 * (len(gen_mtt_edges) - 1)

SYST_PAIRS = [
    ("pileupWeight_up", "pileupWeight_down"),
    ("PreFiringWeight_up", "PreFiringWeight_down"),
    ("eleID_up", "eleID_down"),
    ("btag_sf_up", "btag_sf_down"),
    ("pileUp_sf_up", "pileUp_sf_down"),
]

print(f"Era    : {ERA}")
print(f"Signal : {SIGNAL}")
print(f"Output : {OUT_FILE}")
print(f"Reco File : {RECO_FILE}")
print(f"Gen File: {GEN_FILE}")

# --- LOAD DATA ---
with h5py.File(RECO_FILE, "r") as f:
    best_perm = f["best_perm"][:]
    best_chi2 = f["best_chi2"][:]
    top_lep_px = f["top_lep_px"][:]
    top_lep_py = f["top_lep_py"][:]
    top_lep_pz = f["top_lep_pz"][:]
    top_lep_E = f["top_lep_E"][:]
    top_had_px = f["top_had_px"][:]
    top_had_py = f["top_had_py"][:]
    top_had_pz = f["top_had_pz"][:]
    top_had_E = f["top_had_E"][:]

with h5py.File(GEN_FILE, "r") as f:
    charge = f["charge"][:]
    weights_full = f["weight_variations"][:]
    available_systs = list(weights_full.dtype.names)
    gen_top_pt = f["gen_top_pt"][:]
    gen_top_E = f["gen_top_E"][:]
    gen_top_cosphi = f["gen_top_cosphi"][:]
    gen_top_sinphi = f["gen_top_sinphi"][:]
    gen_top_sinheta = f["gen_top_sinheta"][:]
    gen_antitop_pt = f["gen_antitop_pt"][:]
    gen_antitop_E = f["gen_antitop_E"][:]
    gen_antitop_cosphi = f["gen_antitop_cosphi"][:]
    gen_antitop_sinphi = f["gen_antitop_sinphi"][:]
    gen_antitop_sinheta = f["gen_antitop_sinheta"][:]

Nevt = len(best_perm)
idx = np.arange(Nevt)

# --- DETERMINE RECO SUCCESS (Do not slice arrays) ---
pgof = np.exp(-0.5 * best_chi2)
pass_reco = (best_chi2 > 0) & (pgof > pgof_cut)

lumi_scale = XSEC * LUMI / NGEN
w_nominal = weights_full["nominal"] * lumi_scale

# Safe permutation index to prevent array crashing on failed reco events
safe_perm = np.maximum(best_perm, 0)
charge_best = charge[idx, safe_perm]
cond = charge_best > 0

# --- CALCULATE KINEMATICS FOR ALL EVENTS ---
top_px, antitop_px = assign_top_antitop(cond, top_lep_px, top_had_px)
top_py, antitop_py = assign_top_antitop(cond, top_lep_py, top_had_py)
top_pz, antitop_pz = assign_top_antitop(cond, top_lep_pz, top_had_pz)
top_E, antitop_E = assign_top_antitop(cond, top_lep_E, top_had_E)

mtt_reco = inv_mass(top_px + antitop_px, top_py + antitop_py, top_pz + antitop_pz, top_E + antitop_E)
y_top_reco = rapidity(top_E, top_pz)
y_antitop_reco = rapidity(antitop_E, antitop_pz)

gen_top_px = gen_top_pt[idx, safe_perm] * gen_top_cosphi[idx, safe_perm]
gen_top_py = gen_top_pt[idx, safe_perm] * gen_top_sinphi[idx, safe_perm]
gen_top_pz = gen_top_pt[idx, safe_perm] * gen_top_sinheta[idx, safe_perm]
gen_top_E_sel = gen_top_E[idx, safe_perm]

gen_antitop_px = gen_antitop_pt[idx, safe_perm] * gen_antitop_cosphi[idx, safe_perm]
gen_antitop_py = gen_antitop_pt[idx, safe_perm] * gen_antitop_sinphi[idx, safe_perm]
gen_antitop_pz = gen_antitop_pt[idx, safe_perm] * gen_antitop_sinheta[idx, safe_perm]
gen_antitop_E_sel = gen_antitop_E[idx, safe_perm]

mtt_gen = inv_mass(
    gen_top_px + gen_antitop_px, gen_top_py + gen_antitop_py, 
    gen_top_pz + gen_antitop_pz, gen_top_E_sel + gen_antitop_E_sel
)
y_top_gen = rapidity(gen_top_E_sel, gen_top_pz)
y_antitop_gen = rapidity(gen_antitop_E_sel, gen_antitop_pz)

# --- ASSIGN BINS ---
Nplus_reco, Nminus_reco = make_Nplus_Nminus(y_top_reco, y_antitop_reco)
Nplus_gen, Nminus_gen = make_Nplus_Nminus(y_top_gen, y_antitop_gen)

reco_bin = get_unrolled_bin(mtt_reco, Nplus_reco, Nminus_reco, reco_mtt_edges)
gen_bin = get_unrolled_bin(mtt_gen, Nplus_gen, Nminus_gen, gen_mtt_edges)

# OVERWRITE reco_bin for events that failed the selection cuts
reco_bin[~pass_reco] = -1

# Filter: Keep everything except double-failures
in_matrix = (reco_bin >= 0) | (gen_bin >= 0)

print(f"\n--- Matrix Composition ---")
print(f"Total processed events : {Nevt}")
print(f"Total entering matrix  : {np.sum(in_matrix)}")
print(f" -> Hits   (Gen & Reco valid) : {np.sum((gen_bin >= 0) & (reco_bin >= 0))}")
print(f" -> Misses (Gen valid, Reco failed/out) : {np.sum((gen_bin >= 0) & (reco_bin < 0))}")
print(f" -> Fakes  (Reco valid, Gen out) : {np.sum((gen_bin < 0) & (reco_bin >= 0))}")
print(f"--------------------------\n")

# --- FILL MATRICES ---
fout = ROOT.TFile(OUT_FILE, "RECREATE")

h_nominal = make_response_matrix(
    "response_matrix_nominal", "Nominal response matrix;Gen bin;Reco bin", 
    True, n_gen_bins, n_reco_bins, gen_mtt_edges, reco_mtt_edges
)
fill_response_matrix(h_nominal, gen_bin[in_matrix], reco_bin[in_matrix], w_nominal[in_matrix])
set_matrix_draw_style(h_nominal)
h_nominal.Write()

for up_name, dn_name in SYST_PAIRS:
    source = up_name.replace("_up", "")
    if up_name not in available_systs or dn_name not in available_systs:
        continue

    h_up = make_response_matrix(
        f"response_matrix_{source}Up", f"Response matrix {source} Up;Gen bin;Reco bin", 
        False, n_gen_bins, n_reco_bins, gen_mtt_edges, reco_mtt_edges
    )
    h_dn = make_response_matrix(
        f"response_matrix_{source}Down", f"Response matrix {source} Down;Gen bin;Reco bin", 
        False, n_gen_bins, n_reco_bins, gen_mtt_edges, reco_mtt_edges
    )

    w_up = weights_full[up_name] * lumi_scale
    w_dn = weights_full[dn_name] * lumi_scale

    fill_response_matrix(h_up, gen_bin[in_matrix], reco_bin[in_matrix], w_up[in_matrix])
    fill_response_matrix(h_dn, gen_bin[in_matrix], reco_bin[in_matrix], w_dn[in_matrix])

    zero_bin_errors(h_up)
    zero_bin_errors(h_dn)
    set_matrix_draw_style(h_up)
    set_matrix_draw_style(h_dn)
    h_up.Write()
    h_dn.Write()

fout.Close()
print(f"Done. Wrote {OUT_FILE}")
