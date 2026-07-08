import os
import ctypes
import numpy as np
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptFit(0)

ROOT.gSystem.Load("libUnfold")

# ─────────────────────────────────────────────
# ERA selection — change only this line
# ─────────────────────────────────────────────
ERA    = "2018"
SIGNAL = "ttbar_SemiLeptonic"

# ─────────────────────────────────────────────
# Paths — all files on EOS for SWAN
# ─────────────────────────────────────────────
EOS_DIR = "/eos/user/s/sanskar/tunfold_pipeline"
OUTDIR   = EOS_DIR
OUT_FILE = f"{EOS_DIR}/unfolding_results.root"

os.makedirs(OUTDIR, exist_ok=True)

input_dir = "/eos/user/s/sanskar/tunfold_pipeline"

f_proc = ROOT.TFile.Open("/eos/user/s/sanskar/tunfold_pipeline/unrolled_histograms_nom_and_sys.root")
f_resp = ROOT.TFile.Open("/eos/user/s/sanskar/tunfold_pipeline/response_matrix_tutorial_style_for_2018.root")

h_data = f_proc.Get("ttbar_SemiLeptonic_reco_nominal")
h_truth = f_proc.Get("ttbar_SemiLeptonic_gen_nominal")
h_matrix = f_resp.Get("response_matrix_nominal")

reco_mtt_edges = np.array([300, 375, 450, 525, 600, 675, 750, 825, 900, 975, 1050, 1125, 1200] , dtype=float)
gen_mtt_edges  = np.array([300, 450, 600, 750, 900, 1050, 1200], dtype=float)
n_reco_mtt     = len(reco_mtt_edges) - 1
n_gen_mtt      = len(gen_mtt_edges)  - 1
n_reco_bins    = 2 * n_reco_mtt
n_gen_bins     = 2 * n_gen_mtt


unfold = ROOT.TUnfoldDensity(
    h_matrix,
    ROOT.TUnfold.kHistMapOutputHoriz,
    ROOT.TUnfold.kRegModeCurvature,
    ROOT.TUnfold.kEConstraintArea,
    ROOT.TUnfoldDensity.kDensityModeBinWidth
)

status = unfold.SetInput(h_data)

print(f"SetInput status: {status}")
if status >= 10000:
    raise RuntimeError(
        f"SetInput failed (status={status}) — "
        f"check binning match between h_data and response matrix Y-axis")

    
#=============== Adding systematics===============================
systematics_list = [
    "pileupWeight",
    "PreFiringWeight",
    "eleID",
    "btag_sf",
    "pileUp_sf"
]

for sys_name in systematics_list:
    name_up   = f"response_matrix_{sys_name}Up"
    name_down = f"response_matrix_{sys_name}Down"
    
    # Fetch the matrices
    h_up   = f_resp.Get(name_up)
    h_down = f_resp.Get(name_down)
            
    h_sys_shift = h_up.Clone(f"h_sys_shift_{sys_name}")
    h_sys_shift.SetDirectory(0) 
    
    h_sys_shift.Add(h_down, -1.0)
    h_sys_shift.Scale(0.5)
    unfold.AddSysError(
        h_sys_shift,
        sys_name,                         
        ROOT.TUnfold.kHistMapOutputHoriz, 
        ROOT.TUnfoldSys.kSysErrModeShift  
    )
    
    print(f"  [SUCCESS] Added systematic: {sys_name}")

print("---------------------------------------\n")


#==================================================================


l_curve   = ROOT.TGraph()
log_tau_x = ROOT.TSpline3()
log_tau_y = ROOT.TSpline3()

i_best = unfold.ScanLcurve(100, 0.0, 0.0,
                             l_curve, log_tau_x, log_tau_y)

tau_best = unfold.GetTau()
print(f"  Best scan point : {i_best}")
print(f"  Optimal tau     : {tau_best:.8f}")
print(f"  Chi2 (L)        : {unfold.GetChi2L():.4f}")
print(f"  Chi2 (A)        : {unfold.GetChi2A():.4f}")
print(f"  Rho avg         : {unfold.GetRhoAvg():.4f}")

h_unfolded = unfold.GetOutput("h_unfolded")
h_unfolded.SetDirectory(0)

# Copy gen bin labels from truth
for i in range(1, n_gen_bins + 1):
    h_unfolded.GetXaxis().SetBinLabel(
        i, h_truth.GetXaxis().GetBinLabel(i))

# Total covariance matrix
h_cov = unfold.GetEmatrixTotal("h_cov_total")
h_cov.SetDirectory(0)

# ─────────────────────────────────────────────
# Print unfolded vs truth per bin
# ─────────────────────────────────────────────
print(f"\n{'='*65}")
print("Unfolded vs Truth per gen bin:")
print(f"  {'Bin':<5} {'Label':<28} {'Unfolded':>10} "
      f"{'Err':>10} {'Truth':>10} {'Pull':>8}")
print(f"  {'-'*65}")
for i in range(n_gen_bins):
    unf   = h_unfolded.GetBinContent(i + 1)
    err   = np.sqrt(max(0.0, h_cov.GetBinContent(i+1, i+1)))
    truth = h_truth.GetBinContent(i + 1)
    pull  = (unf - truth) / err if err > 0 else 0.0
    lbl   = h_unfolded.GetXaxis().GetBinLabel(i + 1)
    print(f"  {i+1:<5} {lbl:<28} {unf:>10.2f} "
          f"{err:>10.2f} {truth:>10.2f} {pull:>8.3f}")
    
c_comp = ROOT.TCanvas("c_comp", "Truth vs Unfolded", 900, 700)

# Expand pad1 to fill the whole canvas
pad1 = ROOT.TPad("pad1", "", 0.0, 0.0, 1.0, 1.0)
pad1.SetLeftMargin(0.14)
pad1.SetRightMargin(0.05)
pad1.SetTopMargin(0.12)
pad1.SetBottomMargin(0.25) # Increased bottom margin for x-axis labels
pad1.Draw()
pad1.cd()

h_truth_draw    = h_truth.Clone("h_truth_draw")
h_unfolded_draw = h_unfolded.Clone("h_unfolded_draw")

ymax = max(h_truth_draw.GetMaximum(),
           h_unfolded_draw.GetMaximum()) * 1.35

h_truth_draw.SetLineColor(ROOT.kBlue + 1)
h_truth_draw.SetLineWidth(2)
h_truth_draw.SetFillColorAlpha(ROOT.kBlue + 1, 0.20)
h_truth_draw.GetYaxis().SetTitle("Events")
h_truth_draw.GetYaxis().SetTitleSize(0.055)
h_truth_draw.GetYaxis().SetTitleOffset(1.1)
h_truth_draw.GetYaxis().SetLabelSize(0.045)

# Restore X-axis labels on the main plot
h_truth_draw.GetXaxis().SetLabelSize(0.045)
h_truth_draw.GetXaxis().LabelsOption("v") # Vertical labels, inherited from original ratio pad
h_truth_draw.GetXaxis().SetLabelOffset(0.01)

h_truth_draw.SetMaximum(ymax)
h_truth_draw.SetMinimum(0)
h_truth_draw.Draw("HIST")

h_unfolded_draw.SetMarkerStyle(20)
h_unfolded_draw.SetMarkerSize(1.3)
h_unfolded_draw.SetMarkerColor(ROOT.kBlack)
h_unfolded_draw.SetLineColor(ROOT.kBlack)
h_unfolded_draw.SetLineWidth(2)
h_unfolded_draw.Draw("E1 SAME")

# N+ / N- separator
line = ROOT.TLine(n_gen_mtt, 0, n_gen_mtt, ymax)
line.SetLineColor(ROOT.kRed)
line.SetLineStyle(2)
line.SetLineWidth(2)
line.Draw()

leg = ROOT.TLegend(0.62, 0.68, 0.92, 0.86)
leg.SetBorderSize(0)
leg.SetTextSize(0.045)
leg.AddEntry(h_truth_draw,    "Gen-level truth", "lf")
leg.AddEntry(h_unfolded_draw, "Unfolded",        "lep")
leg.Draw()

lat = ROOT.TLatex()
lat.SetNDC(True)
lat.SetTextSize(0.05)
lat.SetTextFont(62)
lat.DrawLatex(0.16, 0.91, "Closure test  2018")
lat.SetTextFont(42)
lat.SetTextSize(0.04)
lat.DrawLatex(0.16, 0.84, f"#tau = {tau_best:.2e}")

c_comp.cd()
c_comp.SaveAs(os.path.join(OUTDIR, "truth_vs_unfolded.pdf"))
c_comp.SaveAs(os.path.join(OUTDIR, "truth_vs_unfolded.png"))
print(f"\nSaved: truth_vs_unfolded.pdf / .png")
# ─────────────────────────────────────────────
# L-curve canvas
# ─────────────────────────────────────────────
c_lcurve = ROOT.TCanvas("c_lcurve", "L-curve", 700, 600)
c_lcurve.SetLeftMargin(0.14)
c_lcurve.SetRightMargin(0.05)
c_lcurve.SetTopMargin(0.10)
c_lcurve.SetBottomMargin(0.14)

l_curve.SetTitle("L-curve;"
                 "log_{10}(#chi^{2}_{L});"
                 "log_{10}(Regularisation)")
l_curve.SetMarkerStyle(20)
l_curve.SetMarkerSize(0.5)
l_curve.SetMarkerColor(ROOT.kBlue + 1)
l_curve.SetLineColor(ROOT.kBlue + 1)
l_curve.Draw("AL")

# Mark optimal tau point
x_best = ctypes.c_double(0.0)
y_best = ctypes.c_double(0.0)
l_curve.GetPoint(i_best, x_best, y_best)

best_pt = ROOT.TGraph(1)
best_pt.SetPoint(0, x_best.value, y_best.value)
best_pt.SetMarkerStyle(29)
best_pt.SetMarkerSize(2.5)
best_pt.SetMarkerColor(ROOT.kRed)
best_pt.Draw("P SAME")

lat2 = ROOT.TLatex()
lat2.SetNDC(True)
lat2.SetTextSize(0.04)
lat2.DrawLatex(0.16, 0.92, f"Optimal #tau = {tau_best:.2e}")

c_lcurve.SaveAs(os.path.join(OUTDIR, "lcurve.pdf"))
c_lcurve.SaveAs(os.path.join(OUTDIR, "lcurve.png"))
print(f"Saved: lcurve.pdf / .png")

# ─────────────────────────────────────────────
# Save to ROOT file
# ─────────────────────────────────────────────
fout = ROOT.TFile(OUT_FILE, "RECREATE")
h_unfolded.Write("h_unfolded")
h_truth   .Write("h_gen_truth")
h_data    .Write("h_reco_measured")
#h_bkg     .Write("h_bkg")
#h_fakes   .Write("h_fakes")
h_cov     .Write("h_cov_total")
l_curve   .Write("lcurve")
c_comp    .Write("canvas_truth_vs_unfolded")
c_lcurve  .Write("canvas_lcurve")
fout.Close()

print(f"\n{'='*55}")
print(f"Done. Results written to: {OUT_FILE}")
