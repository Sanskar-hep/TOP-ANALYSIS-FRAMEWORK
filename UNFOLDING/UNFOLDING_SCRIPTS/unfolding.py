import os
import ROOT
import ctypes

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptFit(0)

ROOT.gSystem.Load("libUnfold")

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
OUTDIR   = "/eos/user/s/sanskar/UNFOLDING"
OUT_FILE = f"{OUTDIR}/unfolding_results.root"
os.makedirs(OUTDIR, exist_ok=True)

UNROLLED_FILE   = f"{OUTDIR}/unrolled_histograms.root"
BACKGROUND_FILE = f"{OUTDIR}/unrolled_histograms_background.root"
RESPONSE_FILE   = f"{OUTDIR}/response_matrix.root"

n_gen_mtt = 4

# ─────────────────────────────────────────────
# Load input histograms
# ─────────────────────────────────────────────
print("Loading input histograms...")

f_unrolled   = ROOT.TFile(UNROLLED_FILE,   "READ")
f_background = ROOT.TFile(BACKGROUND_FILE, "READ")
f_response   = ROOT.TFile(RESPONSE_FILE,   "READ")

h_data   = f_unrolled.Get("h_reco_ttbar_SemiLeptonic")
h_truth  = f_unrolled.Get("h_gen_ttbar_SemiLeptonic")
h_bkg    = f_background.Get("h_reco_Background")
h_matrix = f_response.Get("response_matrix")

for h, name in [(h_data, "h_data"),
                (h_truth, "h_truth"),
                (h_bkg, "h_bkg"),
                (h_matrix, "response_matrix")]:
    if not h:
        raise RuntimeError(f"Could not load histogram: {name}")

h_data.SetDirectory(0)
h_truth.SetDirectory(0)
h_bkg.SetDirectory(0)
h_matrix.SetDirectory(0)

f_unrolled.Close()
f_background.Close()
f_response.Close()

# ─────────────────────────────────────────────
# Check errors
# ─────────────────────────────────────────────
print("Checking bin errors...")
for i in range(h_data.GetNbinsX()):
    if h_data.GetBinError(i+1) <= 0 and h_data.GetBinContent(i+1) > 0:
        raise RuntimeError(f"Zero error in bin {i+1}")
print("Errors OK.")

# ─────────────────────────────────────────────
# Setup unfolding
# ─────────────────────────────────────────────
unfold = ROOT.TUnfoldDensity(
    h_matrix,
    ROOT.TUnfold.kHistMapOutputHoriz,
    ROOT.TUnfold.kRegModeCurvature,
    ROOT.TUnfold.kEConstraintArea,
    ROOT.TUnfoldDensity.kDensityModeNone  # kDensityModeBinWidth --->Mukund bhaiya
)

status = unfold.SetInput(h_data)
print("SetInput status:", status)

unfold.SubtractBackground(h_bkg, "bkg", 1.0, 0.0)


# ISSUE ----> Error implementation should be checked ---->Mukund bhaiya suggestion 
# ─────────────────────────────────────────────
# L-curve scan
# ─────────────────────────────────────────────
print("Running L-curve scan...")

l_curve   = ROOT.TGraph()
log_tau_x = ROOT.TSpline3()
log_tau_y = ROOT.TSpline3()

i_best = unfold.ScanLcurve(100, 0.0, 0.0,
                           l_curve, log_tau_x, log_tau_y)

tau_best = unfold.GetTau()
print("Best tau:", tau_best)

# ─────────────────────────────────────────────
# Get unfolded histogram
# ─────────────────────────────────────────────
h_unfolded = unfold.GetOutput("h_unfolded")
h_unfolded.SetDirectory(0)

# ─────────────────────────────────────────────
# Truth vs Unfolded overlay
# ─────────────────────────────────────────────
c_comp = ROOT.TCanvas("c_comp", "Truth vs Unfolded", 900, 700)

c_comp.SetLeftMargin(0.12)
c_comp.SetBottomMargin(0.25)

h_truth_draw = h_truth.Clone("h_truth_draw")
h_unfolded_draw = h_unfolded.Clone("h_unfolded_draw")

h_truth_draw.SetLineColor(ROOT.kBlue + 1)
h_truth_draw.SetLineWidth(3)

h_unfolded_draw.SetMarkerStyle(20)
h_unfolded_draw.SetMarkerColor(ROOT.kRed + 1)
h_unfolded_draw.SetLineColor(ROOT.kRed + 1)

ymax = max(h_truth_draw.GetMaximum(), h_unfolded_draw.GetMaximum()) * 1.3

h_truth_draw.SetMaximum(ymax)
h_truth_draw.SetMinimum(0)

h_truth_draw.Draw("HIST")
h_unfolded_draw.Draw("E1 SAME")

# vertical separator
line = ROOT.TLine(n_gen_mtt, 0, n_gen_mtt, ymax)
line.SetLineStyle(2)
line.Draw()

# legend
leg = ROOT.TLegend(0.65, 0.75, 0.93, 0.90)
leg.SetBorderSize(0)
leg.SetFillStyle(0)
leg.AddEntry(h_truth_draw, "Truth", "L")
leg.AddEntry(h_unfolded_draw, "Unfolded", "PE")
leg.Draw()

c_comp.SaveAs(f"{OUTDIR}/truth_vs_unfolded.png")

# ─────────────────────────────────────────────
# L-curve plot (FIXED)
# ─────────────────────────────────────────────
c_lcurve = ROOT.TCanvas("c_lcurve", "L-curve", 700, 600)

l_curve.Draw("AL")

x_best = ctypes.c_double(0.)
y_best = ctypes.c_double(0.)

l_curve.GetPoint(i_best, x_best, y_best)

x = x_best.value
y = y_best.value

best = ROOT.TGraph(1)
best.SetPoint(0, x, y)
best.SetMarkerStyle(29)
best.SetMarkerColor(ROOT.kRed)
best.SetMarkerSize(2)
best.Draw("P SAME")

latex = ROOT.TLatex()
latex.DrawLatex(x, y, f"#tau = {tau_best:.2e}")

c_lcurve.SaveAs(f"{OUTDIR}/lcurve.png")

# ─────────────────────────────────────────────
# Save everything to ROOT file
# ─────────────────────────────────────────────
fout = ROOT.TFile(OUT_FILE, "RECREATE")

h_unfolded.Write("h_unfolded")
h_truth.Write("h_truth")
h_data.Write("h_data")
h_bkg.Write("h_bkg")
l_curve.Write("l_curve")

# Save canvas
c_comp.Write("canvas_truth_vs_unfolded")
c_lcurve.Write("canvas_lcurve")

fout.Close()

print("Done. Everything saved.")
