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
        "reco_mtt_edges": np.array(
            cfg["binning"]["reco"]["mtt_edges"], dtype=float),
        "gen_mtt_edges": np.array(
            cfg["binning"]["gen"]["mtt_edges"], dtype=float),
        "signal": signal,
        "xsec": float(signal_cfg["xsec"][era]),
        "ngen": int(signal_cfg["ngen"][era]),
    }


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

RECO_FILE = os.path.join(
    cfg["reco_dir"], f"{SIGNAL}{cfg['file_suffix']}")
GEN_FILE = os.path.join(
    cfg["gen_dir"], f"{SIGNAL}{cfg['gen_suffix']}")

OUTDIR = os.path.join(SCRIPT_DIR, "results_response_matrix", ERA)
OUT_FILE = os.path.join(
    OUTDIR, f"response_matrix_tutorial_style_for_{ERA}.root")
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
            labels.append(
                f"{sign}({int(mtt_edges[i])}-{int(mtt_edges[i + 1])})"
            )
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


def make_response_matrix(name, title, with_stat_errors):
    h = ROOT.TH2D(
        name, title,
        n_gen_bins, 0, n_gen_bins,
        n_reco_bins, 0, n_reco_bins,
    )
    if with_stat_errors:
        h.Sumw2()

    for i, label in enumerate(make_labels(gen_mtt_edges)):
        h.GetXaxis().SetBinLabel(i + 1, label)
    for i, label in enumerate(make_labels(reco_mtt_edges)):
        h.GetYaxis().SetBinLabel(i + 1, label)

    h.SetMinimum(0.0)
    h.SetOption("COLZ0")
    return h


def fill_response_matrix(h, gen_bin, reco_bin, weights):
    for g, r, w in zip(gen_bin, reco_bin, weights):
        h.Fill(float(g) + 0.5, float(r) + 0.5, float(w))


def zero_bin_errors(h):
    for gx in range(1, h.GetNbinsX() + 1):
        for ry in range(1, h.GetNbinsY() + 1):
            h.SetBinError(gx, ry, 0.0)


def set_matrix_draw_style(h):
    zmax = h.GetMaximum()
    # ROOT can leave exactly empty bins unpainted. A tiny negative minimum
    # plus COLZ0 makes zero-content cells use the low end of the palette.
    h.SetMinimum(-0.01 * max(1.0, zmax))
    h.SetMaximum(zmax)
    h.SetOption("COLZ0")


print(f"Era    : {ERA}")
print(f"Signal : {SIGNAL}")
print(f"Reco   : {RECO_FILE}")
print(f"Gen    : {GEN_FILE}")
print(f"Output : {OUT_FILE}")

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

if len(best_perm) != len(weights_full) or len(best_perm) != charge.shape[0]:
    raise RuntimeError(
        "Reco and gen/weight arrays do not have matching first dimension: "
        f"best_perm={len(best_perm)}, weights={len(weights_full)}, "
        f"charge={charge.shape[0]}"
    )

Nevt = len(best_perm)
idx = np.arange(Nevt)
#valid_perm = (best_perm >= 0) & (best_perm < charge.shape[1])
pgof = np.exp(-0.5 * best_chi2)
event_mask =(best_chi2 > 0) & (pgof > pgof_cut)

idx_sel = idx[event_mask]
best_perm_sel = best_perm[event_mask]
weights_sel = weights_full[event_mask]
lumi_scale = XSEC * LUMI / NGEN
w_nominal = weights_sel["nominal"] * lumi_scale

print(f"Total events          : {Nevt}")
#print(f"Valid best_perm       : {np.sum(valid_perm)}")
print(f"After reco pgof mask  : {np.sum(event_mask)}")

charge_best = charge[idx_sel, best_perm_sel]
cond = charge_best > 0

top_px, antitop_px = assign_top_antitop(
    cond, top_lep_px[event_mask], top_had_px[event_mask])
top_py, antitop_py = assign_top_antitop(
    cond, top_lep_py[event_mask], top_had_py[event_mask])
top_pz, antitop_pz = assign_top_antitop(
    cond, top_lep_pz[event_mask], top_had_pz[event_mask])
top_E, antitop_E = assign_top_antitop(
    cond, top_lep_E[event_mask], top_had_E[event_mask])

mtt_reco = inv_mass(
    top_px + antitop_px,
    top_py + antitop_py,
    top_pz + antitop_pz,
    top_E + antitop_E,
)
y_top_reco = rapidity(top_E, top_pz)
y_antitop_reco = rapidity(antitop_E, antitop_pz)

gen_top_px = gen_top_pt[idx_sel, best_perm_sel] * gen_top_cosphi[idx_sel, best_perm_sel]
gen_top_py = gen_top_pt[idx_sel, best_perm_sel] * gen_top_sinphi[idx_sel, best_perm_sel]
gen_top_pz = gen_top_pt[idx_sel, best_perm_sel] * gen_top_sinheta[idx_sel, best_perm_sel]
gen_top_E_sel = gen_top_E[idx_sel, best_perm_sel]

gen_antitop_px = (
    gen_antitop_pt[idx_sel, best_perm_sel]
    * gen_antitop_cosphi[idx_sel, best_perm_sel]
)
gen_antitop_py = (
    gen_antitop_pt[idx_sel, best_perm_sel]
    * gen_antitop_sinphi[idx_sel, best_perm_sel]
)
gen_antitop_pz = (
    gen_antitop_pt[idx_sel, best_perm_sel]
    * gen_antitop_sinheta[idx_sel, best_perm_sel]
)
gen_antitop_E_sel = gen_antitop_E[idx_sel, best_perm_sel]

mtt_gen = inv_mass(
    gen_top_px + gen_antitop_px,
    gen_top_py + gen_antitop_py,
    gen_top_pz + gen_antitop_pz,
    gen_top_E_sel + gen_antitop_E_sel,
)
y_top_gen = rapidity(gen_top_E_sel, gen_top_pz)
y_antitop_gen = rapidity(gen_antitop_E_sel, gen_antitop_pz)

Nplus_reco, Nminus_reco = make_Nplus_Nminus(y_top_reco, y_antitop_reco)
Nplus_gen, Nminus_gen = make_Nplus_Nminus(y_top_gen, y_antitop_gen)

reco_bin = get_unrolled_bin(
    mtt_reco, Nplus_reco, Nminus_reco, reco_mtt_edges)
gen_bin = get_unrolled_bin(
    mtt_gen, Nplus_gen, Nminus_gen, gen_mtt_edges)

# Tutorial-style response matrix: keep only events with both bins defined.
in_matrix = (reco_bin >= 0) & (gen_bin >= 0)

print(f"Events entering matrix: {np.sum(in_matrix)}")
print(f"Excluded by reco/gen bin requirement: {np.sum(~in_matrix)}")

h_nominal = make_response_matrix(
    "response_matrix_nominal",
    "Nominal response matrix;Gen bin;Reco bin",
    with_stat_errors=True,
)
fill_response_matrix(
    h_nominal,
    gen_bin[in_matrix],
    reco_bin[in_matrix],
    w_nominal[in_matrix],
)
set_matrix_draw_style(h_nominal)

fout = ROOT.TFile(OUT_FILE, "RECREATE")
h_nominal.Write()

for up_name, dn_name in SYST_PAIRS:
    source = up_name.replace("_up", "")

    if up_name not in available_systs or dn_name not in available_systs:
        print(f"[SKIP] {source}: missing Up/Down weights")
        continue

    h_up = make_response_matrix(
        f"response_matrix_{source}Up",
        f"Response matrix {source} Up;Gen bin;Reco bin",
        with_stat_errors=False,
    )
    h_dn = make_response_matrix(
        f"response_matrix_{source}Down",
        f"Response matrix {source} Down;Gen bin;Reco bin",
        with_stat_errors=False,
    )

    w_up = weights_sel[up_name] * lumi_scale
    w_dn = weights_sel[dn_name] * lumi_scale

    fill_response_matrix(
        h_up, gen_bin[in_matrix], reco_bin[in_matrix], w_up[in_matrix])
    fill_response_matrix(
        h_dn, gen_bin[in_matrix], reco_bin[in_matrix], w_dn[in_matrix])

    zero_bin_errors(h_up)
    zero_bin_errors(h_dn)
    set_matrix_draw_style(h_up)
    set_matrix_draw_style(h_dn)
    h_up.Write()
    h_dn.Write()
    print(f"Wrote systematic pair: {source} Up/Down")

canvas = ROOT.TCanvas("canvas_response_matrix_nominal", "Response Matrix", 1000, 800)
canvas.SetLeftMargin(0.22)
canvas.SetBottomMargin(0.22)
canvas.SetRightMargin(0.14)
ROOT.gStyle.SetPalette(ROOT.kBird)
h_nominal.GetXaxis().LabelsOption("v")
h_nominal.GetXaxis().SetLabelSize(0.028)
h_nominal.GetYaxis().SetLabelSize(0.028)
h_nominal.GetZaxis().SetTitle("Weighted events")
h_nominal.Draw("COLZ0")
canvas.Write()
canvas.SaveAs(os.path.join(OUTDIR, "response_matrix_tutorial_style.pdf"))
canvas.SaveAs(os.path.join(OUTDIR, "response_matrix_tutorial_style.png"))

fout.Close()

print(f"Done. Wrote {OUT_FILE}")
print("Nominal bin errors are MC statistical errors from Sumw2.")
print("Systematic Up/Down matrices contain shifted counts with bin errors set to zero.")
