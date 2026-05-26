import os
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptFit(0)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
OUTDIR   = "/home/irfan/sanskar/outputs/Unfolding_allEras/2018"
IN_FILE  = f"{OUTDIR}/unrolled_histograms.root"
OUT_FILE = f"{OUTDIR}/unrolled_histograms_background.root"

os.makedirs(OUTDIR, exist_ok=True)

# ─────────────────────────────────────────────
# Switch: set to True to clip negative bins to
# zero, False to keep negative values as-is
# ─────────────────────────────────────────────
CLIP_NEGATIVE = False

# All backgrounds = everything except ttbar_SemiLeptonic
BACKGROUND_DATASETS = [
    "ttbar_FullyLeptonic",
    "Tchannel", "Tbarchannel", "Schannel", "tw_top", "tw_antitop",
    "WJetsToLNu_0J", "WJetsToLNu_1J", "WJetsToLNu_2J",
    "DYJetsToLL",
    "WWTo2L2Nu", "WZTo2Q2L", "ZZTo2L2Nu", "ZZTo2Q2L",
]

n_reco_mtt = 8   # 16 unrolled reco bins

# ─────────────────────────────────────────────
# Helper: draw styled canvas and write to file
# ─────────────────────────────────────────────
def write_canvas(fout, h, n_mtt, outname):
    cname  = os.path.basename(outname)
    n_bins = h.GetNbinsX()
    c = ROOT.TCanvas(cname, cname, max(1400, 100 * n_bins), 700)
    c.SetLeftMargin(0.10)
    c.SetRightMargin(0.04)
    c.SetBottomMargin(0.38)
    c.SetTopMargin(0.10)

    h.SetLineColor(ROOT.kBlack)
    h.SetLineWidth(2)
    h.SetFillColor(ROOT.kGray + 1)
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

    # dashed vertical line separating N+ / N-
    line = ROOT.TLine(n_mtt, 0, n_mtt, ymax)
    line.SetLineColor(ROOT.kRed)
    line.SetLineStyle(2)
    line.SetLineWidth(2)
    line.Draw("same")

    # N+ / N- region labels
    latex = ROOT.TLatex()
    latex.SetNDC(True)
    latex.SetTextSize(0.05)
    latex.SetTextColor(ROOT.kRed + 1)
    latex.SetTextAlign(22)
    latex.DrawLatex(0.10 + 0.86 * 0.25, 0.93, "N_{+}")
    latex.DrawLatex(0.10 + 0.86 * 0.75, 0.93, "N_{-}")

    # background label top-left
    label_latex = ROOT.TLatex()
    label_latex.SetNDC(True)
    label_latex.SetTextSize(0.045)
    label_latex.SetTextColor(ROOT.kBlack)
    label_latex.SetTextAlign(11)
    label_latex.DrawLatex(0.12, 0.93, "Total Background")

    c.RedrawAxis()
    c.Update()
    c.SaveAs(f"{outname}.pdf")
    c.SaveAs(f"{outname}.png")
    print(f"  Saved: {outname}.pdf / .png")

    fout.cd()
    h.Write()                      # plain TH1D
    c.Write(f"canvas_{cname}")     # styled TCanvas

# ─────────────────────────────────────────────
# Open input and output ROOT files
# ─────────────────────────────────────────────
fin  = ROOT.TFile(IN_FILE,  "READ")
fout = ROOT.TFile(OUT_FILE, "RECREATE")

if fin.IsZombie():
    raise RuntimeError(f"Cannot open input ROOT file: {IN_FILE}")

print(f"Reading from : {IN_FILE}")
print(f"Writing to   : {OUT_FILE}\n")
print(f"{'='*55}")
print(f"Summing {len(BACKGROUND_DATASETS)} background datasets:")

# ─────────────────────────────────────────────
# Sum all background histograms
# ─────────────────────────────────────────────
h_bkg = None
missing = []

for dataset in BACKGROUND_DATASETS:
    hname = f"h_reco_{dataset}"
    h_src = fin.Get(hname)

    if not h_src:
        print(f"  WARNING: {hname} not found - skipping")
        missing.append(dataset)
        continue

    h_clone = h_src.Clone(f"_tmp_{dataset}")
    h_clone.SetDirectory(0)

    if h_bkg is None:
        h_bkg = h_clone.Clone("h_reco_Background")
        h_bkg.SetDirectory(0)
        print(f"  + {dataset} (base)")
    else:
        h_bkg.Add(h_clone)
        print(f"  + {dataset}")

if h_bkg is None:
    raise RuntimeError("No background histograms found - check input ROOT file.")

if missing:
    print(f"\n  Missing datasets: {missing}")

h_bkg.SetTitle("Total Background Reco Unrolled;Bin;Events")

# ─────────────────────────────────────────────
# Clip negative bins to zero (optional)
# (can arise from NLO negative weights in low-stats bins e.g. DYJetsToLL)
# ─────────────────────────────────────────────
print(f"\n{'='*55}")
if CLIP_NEGATIVE:
    print("Checking for negative bins (CLIP_NEGATIVE=True):")
    any_negative = False
    for i in range(h_bkg.GetNbinsX()):
        c = h_bkg.GetBinContent(i + 1)
        if c < 0:
            lbl = h_bkg.GetXaxis().GetBinLabel(i + 1)
            print(f"  WARNING: bin {i+1} ({lbl}) = {c:.4f} → clipping to 0")
            h_bkg.SetBinContent(i + 1, 0.0)
            any_negative = True
    if not any_negative:
        print("  No negative bins found.")
else:
    print("Negative bin clipping is OFF (CLIP_NEGATIVE=False) — keeping values as-is.")
    for i in range(h_bkg.GetNbinsX()):
        c = h_bkg.GetBinContent(i + 1)
        if c < 0:
            lbl = h_bkg.GetXaxis().GetBinLabel(i + 1)
            print(f"  INFO: bin {i+1} ({lbl}) = {c:.4f} (kept)")

# ─────────────────────────────────────────────
# Print per-bin summary
# ─────────────────────────────────────────────
n_bins = h_bkg.GetNbinsX()
print(f"\n{'='*55}")
print(f"Total background unrolled counts ({n_bins} bins):")
print(f"{'Bin':<6} {'Label':<28} {'Content':>12}")
print("-" * 48)
for i in range(n_bins):
    lbl = h_bkg.GetXaxis().GetBinLabel(i + 1)
    c   = h_bkg.GetBinContent(i + 1)
    print(f"  {i+1:<4} {lbl:<28} {c:>12.4f}")

# ─────────────────────────────────────────────
# Draw, save PNG/PDF, write to ROOT file
# ─────────────────────────────────────────────
print(f"\n{'='*55}")
write_canvas(fout, h_bkg, n_reco_mtt, f"{OUTDIR}/unrolled_reco_Background")

# ─────────────────────────────────────────────
# Close files
# ─────────────────────────────────────────────
fin.Close()
fout.Close()

print(f"\n{'='*55}")
print(f"Done.")
print(f"  h_reco_Background                       -> plain TH1D (16 bins)")
print(f"  canvas_unrolled_reco_Background         -> styled TCanvas")
print(f"  ROOT file: {OUT_FILE}")
