import os
import h5py
import numpy as np
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptFit(0)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
Y_0    = 1.2
LUMI   = 59222.7416  # pb-1, 2018

OUTDIR = "/home/irfan/sanskar/outputs/Unfolding_allEras/2018"
os.makedirs(OUTDIR, exist_ok=True)

RECO_DIR = "/home/irfan/sanskar/outputs/2018_reco/RECO_REVISIT"
GEN_DIR  = "/nfs/home/sanskar/2018Analysis/RECO/minimizer-script/NEW_THINGS"

# {dataset: [xsec (pb), ngen]}
UL2018 = {
    "ttbar_SemiLeptonic"   : [366.3,     472977862 ],
    "ttbar_FullyLeptonic"  : [88.5,      143830836 ],
    "Tchannel"             : [134.2,     166637158 ],
    "Tbarchannel"          : [80.0,      89985007  ],
    "Schannel"             : [2.215836,  12444591  ],
    "tw_top"               : [39.65,     11270430  ],
    "tw_antitop"           : [39.65,     10949620  ],
    "DYJetsToLL"           : [6424.0,    129037134 ],
    "WJetsToLNu_0J"        : [52780.0,   137259710 ],
    "WJetsToLNu_1J"        : [8832.0,    87594835  ],
    "WJetsToLNu_2J"        : [3276.0,    29028341  ],
    "WWTo2L2Nu"            : [11.09,     9962019   ],
    "WZTo2Q2L"             : [6.565,     17952068  ],
    "ZZTo2L2Nu"            : [0.974,     16826232  ],
    "ZZTo2Q2L"             : [3.676,     19082659  ],
}

# reco binning: 8 mtt bins × [N+, N-] = 16 bins
reco_mtt_edges = np.array([300, 450, 600, 750, 900, 1050, 1200, 1350, 1500], dtype=float)
n_reco_mtt     = len(reco_mtt_edges) - 1   # 8

# gen binning (ttbar_SemiLeptonic only): 4 mtt bins × [N+, N-] = 8 bins
gen_mtt_edges  = np.array([300, 600, 900, 1200, 1500], dtype=float)
n_gen_mtt      = len(gen_mtt_edges) - 1    # 4

# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────
def inv_mass(px, py, pz, E):
    return np.sqrt(np.maximum(E**2 - (px**2 + py**2 + pz**2), 0.0))

def rapidity(E, pz):
    return 0.5 * np.log((E + pz) / (E - pz))

def assign_top_antitop(cond, lep, had):
    return np.where(cond, lep, had), np.where(cond, had, lep)

def make_Nplus_Nminus(y_top, y_antitop, Y_0, label=""):
    """Mutually exclusive N+/N- masks."""
    top_fwd     = np.abs(y_top)     > Y_0
    antitop_fwd = np.abs(y_antitop) > Y_0
    Nplus  = top_fwd  & ~antitop_fwd
    Nminus = antitop_fwd & ~top_fwd
    n_plus, n_minus   = np.sum(Nplus), np.sum(Nminus)
    n_both    = np.sum( top_fwd &  antitop_fwd)
    n_neither = np.sum(~top_fwd & ~antitop_fwd)
    n_total   = n_plus + n_minus + n_both + n_neither
    print(f"  [{label}] N+={n_plus}  N-={n_minus}  "
          f"both={n_both}({100*n_both/n_total:.1f}%)  "
          f"neither={n_neither}({100*n_neither/n_total:.1f}%)")
    if n_plus + n_minus > 0:
        print(f"  [{label}] Inclusive A_out = {(n_plus-n_minus)/(n_plus+n_minus):.4f}")
    return Nplus, Nminus

def fill_unrolled(mtt, Nplus, Nminus, mtt_edges, w):
    """Weighted counts per unrolled bin."""
    n_mtt  = len(mtt_edges) - 1
    counts = np.zeros(2 * n_mtt, dtype=float)
    for i in range(n_mtt):
        lo, hi          = mtt_edges[i], mtt_edges[i + 1]
        in_mtt          = (mtt >= lo) & (mtt < hi)
        counts[i]       = np.sum(w[in_mtt & Nplus])
        counts[n_mtt+i] = np.sum(w[in_mtt & Nminus])
    return counts

def fill_errors(mtt, Nplus, Nminus, mtt_edges, w):
    """sqrt(sum w^2) per unrolled bin — correct stat error for weighted hists."""
    n_mtt  = len(mtt_edges) - 1
    errors = np.zeros(2 * n_mtt, dtype=float)
    for i in range(n_mtt):
        lo, hi          = mtt_edges[i], mtt_edges[i + 1]
        in_mtt          = (mtt >= lo) & (mtt < hi)
        errors[i]       = np.sqrt(np.sum(w[in_mtt & Nplus]  ** 2))
        errors[n_mtt+i] = np.sqrt(np.sum(w[in_mtt & Nminus] ** 2))
    return errors

def make_labels(mtt_edges):
    n, lbl = len(mtt_edges) - 1, []
    for sign in ["N_{+}", "N_{-}"]:
        for i in range(n):
            lbl.append(f"{sign}({int(mtt_edges[i])}-{int(mtt_edges[i+1])})")
    return lbl

def fill_th1(name, title, counts, errors, labels):
    n = len(counts)
    h = ROOT.TH1D(name, title, n, 0, n)
    h.Sumw2()
    for i, (c, e) in enumerate(zip(counts, errors)):
        h.SetBinContent(i + 1, c)
        h.SetBinError(i + 1, e)
    for i, lbl in enumerate(labels):
        h.GetXaxis().SetBinLabel(i + 1, lbl)
    return h

def write_to_rootfile(fout, h, n_mtt, outname):
    """Save styled TCanvas + plain TH1D to ROOT file, and PNG/PDF to disk."""
    cname  = os.path.basename(outname)
    n_bins = h.GetNbinsX()
    c = ROOT.TCanvas(cname, cname, max(1400, 100 * n_bins), 700)
    c.SetLeftMargin(0.10)
    c.SetRightMargin(0.04)
    c.SetBottomMargin(0.38)
    c.SetTopMargin(0.10)

    h.SetLineColor(ROOT.kBlack)
    h.SetLineWidth(2)
    h.SetFillColor(ROOT.kAzure + 7)
    h.SetFillStyle(1001)
    h.GetXaxis().SetLabelSize(0.038)
    h.GetXaxis().SetLabelOffset(0.005)
    h.GetXaxis().LabelsOption("v")
    h.GetYaxis().SetTitle("Events")
    h.GetYaxis().SetTitleSize(0.05)
    h.GetYaxis().SetTitleOffset(0.9)
    h.GetYaxis().SetLabelSize(0.04)

    ymax = h.GetMaximum() * 1.20
    h.SetMinimum(0)
    h.SetMaximum(ymax)
    h.Draw("BAR")

    line = ROOT.TLine(n_mtt, 0, n_mtt, ymax)
    line.SetLineColor(ROOT.kRed)
    line.SetLineStyle(2)
    line.SetLineWidth(2)
    line.Draw("same")

    latex = ROOT.TLatex()
    latex.SetNDC(True)
    latex.SetTextSize(0.05)
    latex.SetTextColor(ROOT.kRed + 1)
    latex.SetTextAlign(22)
    latex.DrawLatex(0.10 + 0.86 * 0.25, 0.93, "N_{+}")
    latex.DrawLatex(0.10 + 0.86 * 0.75, 0.93, "N_{-}")

    c.RedrawAxis()
    c.Update()

    # save PNG/PDF to disk
    c.SaveAs(f"{outname}.pdf")
    c.SaveAs(f"{outname}.png")
    print(f"  Saved: {outname}.pdf / .png")

    # write both canvas and histogram to ROOT file
    fout.cd()
    h.Write()                              # plain TH1D for TUnfold
    c.Write(f"canvas_{cname}")            # styled TCanvas for display


# ─────────────────────────────────────────────
# Open output ROOT file (all histograms go here)
# ─────────────────────────────────────────────
fout = ROOT.TFile(f"{OUTDIR}/unrolled_histograms.root", "RECREATE")

reco_labels = make_labels(reco_mtt_edges)
gen_labels  = make_labels(gen_mtt_edges)

# ─────────────────────────────────────────────
# Loop over all datasets
# ─────────────────────────────────────────────
for dataset, (xsec, ngen) in UL2018.items():
    print(f"\n{'='*60}")
    print(f"Processing: {dataset}")
    print(f"  xsec={xsec} pb  ngen={ngen}  lumi_wt={xsec*LUMI/ngen:.6f}")

    reco_file = f"{RECO_DIR}/{dataset}_reco_variables_updated_fit_2018.h5"
    gen_file  = f"{GEN_DIR}/{dataset}_reco_variables_updated.h5"

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

    # ── Load gen file (weights always; kinematics only for ttbar_SemiLeptonic) ──
    with h5py.File(gen_file, "r") as f:
        charge  = f["charge"][:]
        weights = f["weights"][:]

        if dataset == "ttbar_SemiLeptonic":
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

    # ── Lumi-normalized weight ──────────────────────────
    weight_lumi_norm = weights * (xsec * LUMI) / ngen

    # ── Common: event index, best-perm charge, pgof mask ──
    Nevt        = best_perm.shape[0]
    idx         = np.arange(Nevt)
    charge_best = charge[idx, best_perm]

    pgof = np.exp(-0.5 * best_chi2)
    mask = (pgof > 0.2) & (best_chi2 > 0)

    # ── RECO: assign top/antitop ────────────────────────
    cond = charge_best > 0
    top_px, antitop_px = assign_top_antitop(cond, top_lep_px, top_had_px)
    top_py, antitop_py = assign_top_antitop(cond, top_lep_py, top_had_py)
    top_pz, antitop_pz = assign_top_antitop(cond, top_lep_pz, top_had_pz)
    top_E,  antitop_E  = assign_top_antitop(cond, top_lep_E,  top_had_E)

    mtt_reco_full       = inv_mass(top_px + antitop_px,
                                   top_py + antitop_py,
                                   top_pz + antitop_pz,
                                   top_E  + antitop_E)
    y_top_reco_full     = rapidity(top_E,     top_pz)
    y_antitop_reco_full = rapidity(antitop_E, antitop_pz)

    mtt_reco       = mtt_reco_full[mask]
    y_top_reco     = y_top_reco_full[mask]
    y_antitop_reco = y_antitop_reco_full[mask]
    weights_reco   = weight_lumi_norm[mask]

    # ── RECO N+/N- and unrolled histogram ───────────────
    print(f"  --- Reco ---")
    Nplus_reco, Nminus_reco = make_Nplus_Nminus(y_top_reco, y_antitop_reco, Y_0, label="Reco")

    counts_reco = fill_unrolled(mtt_reco, Nplus_reco, Nminus_reco, reco_mtt_edges, weights_reco)
    errors_reco = fill_errors(  mtt_reco, Nplus_reco, Nminus_reco, reco_mtt_edges, weights_reco)

    h_reco = fill_th1(f"h_reco_{dataset}",
                      f"{dataset} Reco Unrolled;Bin;Events",
                      counts_reco, errors_reco, reco_labels)

    write_to_rootfile(fout, h_reco, n_reco_mtt, f"{OUTDIR}/unrolled_reco_{dataset}")

    # ── GEN: only for ttbar_SemiLeptonic ────────────────
    if dataset == "ttbar_SemiLeptonic":
        gen_top_px_best     = gen_top_pt[idx, best_perm]     * gen_top_cosphi[idx, best_perm]
        gen_top_py_best     = gen_top_pt[idx, best_perm]     * gen_top_sinphi[idx, best_perm]
        gen_top_pz_best     = gen_top_pt[idx, best_perm]     * gen_top_sinheta[idx, best_perm]
        gen_top_E_best      = gen_top_E_arr[idx, best_perm]

        gen_antitop_px_best = gen_antitop_pt[idx, best_perm] * gen_antitop_cosphi[idx, best_perm]
        gen_antitop_py_best = gen_antitop_pt[idx, best_perm] * gen_antitop_sinphi[idx, best_perm]
        gen_antitop_pz_best = gen_antitop_pt[idx, best_perm] * gen_antitop_sinheta[idx, best_perm]
        gen_antitop_E_best  = gen_antitop_E_arr[idx, best_perm]

        mtt_gen_full       = inv_mass(gen_top_px_best     + gen_antitop_px_best,
                                      gen_top_py_best     + gen_antitop_py_best,
                                      gen_top_pz_best     + gen_antitop_pz_best,
                                      gen_top_E_best      + gen_antitop_E_best)
        y_top_gen_full     = rapidity(gen_top_E_best,     gen_top_pz_best)
        y_antitop_gen_full = rapidity(gen_antitop_E_best, gen_antitop_pz_best)

        mtt_gen       = mtt_gen_full[mask]
        y_top_gen     = y_top_gen_full[mask]
        y_antitop_gen = y_antitop_gen_full[mask]
        weights_gen   = weight_lumi_norm[mask]

        print(f"  --- Gen ---")
        Nplus_gen, Nminus_gen = make_Nplus_Nminus(y_top_gen, y_antitop_gen, Y_0, label="Gen")

        counts_gen = fill_unrolled(mtt_gen, Nplus_gen, Nminus_gen, gen_mtt_edges, weights_gen)
        errors_gen = fill_errors(  mtt_gen, Nplus_gen, Nminus_gen, gen_mtt_edges, weights_gen)

        h_gen = fill_th1("h_gen_ttbar_SemiLeptonic",
                         "ttbar_SemiLeptonic Gen Unrolled;Bin;Events",
                         counts_gen, errors_gen, gen_labels)

        write_to_rootfile(fout, h_gen, n_gen_mtt, f"{OUTDIR}/unrolled_gen_ttbar_SemiLeptonic")

# ─────────────────────────────────────────────
# Close ROOT file
# ─────────────────────────────────────────────
fout.Close()
print(f"\n{'='*60}")
print(f"All done. ROOT file: {OUTDIR}/unrolled_histograms.root")
