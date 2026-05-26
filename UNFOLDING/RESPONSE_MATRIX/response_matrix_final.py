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
Y_0  = 1.2
LUMI = 59222.7416          # pb-1, 2018

XSEC = 366.3               # ttbar_SemiLeptonic [pb]
NGEN = 472977862

OUTDIR   = "/home/irfan/sanskar/outputs/Unfolding_allEras/2018"
OUT_FILE = f"{OUTDIR}/response_matrix.root"
os.makedirs(OUTDIR, exist_ok=True)

RECO_FILE = "/home/irfan/sanskar/outputs/2018_reco/RECO_REVISIT/ttbar_SemiLeptonic_reco_variables_updated_fit_2018.h5"
GEN_FILE  = "/nfs/home/sanskar/2018Analysis/RECO/minimizer-script/NEW_THINGS/ttbar_SemiLeptonic_reco_variables_updated.h5"

# ─────────────────────────────────────────────
# Binning
# reco: 8 mtt bins x [N+, N-] = 16 bins
#   bins 0..7  → N+ (|y_t| > Y0, |y_tbar| <= Y0)
#   bins 8..15 → N- (|y_tbar| > Y0, |y_t| <= Y0)
# gen:  4 mtt bins x [N+, N-] = 8 bins
#   bins 0..3  → N+
#   bins 4..7  → N-
# ─────────────────────────────────────────────
reco_mtt_edges = np.array([300, 450, 600, 750, 900, 1050, 1200, 1350, 1500], dtype=float)
gen_mtt_edges  = np.array([300, 600, 900, 1200, 1500], dtype=float)

n_reco_mtt  = len(reco_mtt_edges) - 1   # 8
n_gen_mtt   = len(gen_mtt_edges)  - 1   # 4
n_reco_bins = 2 * n_reco_mtt            # 16
n_gen_bins  = 2 * n_gen_mtt             # 8

# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────
def inv_mass(px, py, pz, E):
    """Invariant mass, clamped to zero for numerical safety."""
    return np.sqrt(np.maximum(E**2 - (px**2 + py**2 + pz**2), 0.0))

def rapidity(E, pz):
    """Longitudinal rapidity."""
    return 0.5 * np.log((E + pz) / (E - pz))

def assign_top_antitop(cond, lep, had):
    """Assign top/antitop based on lepton charge sign."""
    return np.where(cond, lep, had), np.where(cond, had, lep)

def make_labels(mtt_edges):
    """Make unrolled bin labels: N+ bins first, then N- bins."""
    n, lbl = len(mtt_edges) - 1, []
    for sign in ["N_{+}", "N_{-}"]:
        for i in range(n):
            lbl.append(f"{sign}({int(mtt_edges[i])}-{int(mtt_edges[i+1])})")
    return lbl

def get_unrolled_bin(mtt, is_Nplus, is_Nminus, mtt_edges):
    """
    Assign each event to an unrolled bin index (0-based).

    Layout:
      bins 0 .. n_mtt-1      → N+ events in each mtt slice
      bins n_mtt .. 2*n_mtt-1 → N- events in each mtt slice

    Returns -1 for events that:
      - fail the mutually exclusive N+/N- condition (both or neither forward)
      - have mtt outside the defined edges
    """
    n_mtt  = len(mtt_edges) - 1
    result = np.full(len(mtt), -1, dtype=int)
    for i in range(n_mtt):
        lo, hi = mtt_edges[i], mtt_edges[i + 1]
        in_mtt = (mtt >= lo) & (mtt < hi)
        result[in_mtt & is_Nplus]  = i
        result[in_mtt & is_Nminus] = n_mtt + i
    return result

# ─────────────────────────────────────────────
# Load reco file
# ─────────────────────────────────────────────
print("Loading reco file...")
with h5py.File(RECO_FILE, "r") as f:
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

# ─────────────────────────────────────────────
# Load gen file
# ─────────────────────────────────────────────
print("Loading gen file...")
with h5py.File(GEN_FILE, "r") as f:
    charge              = f["charge"][:]
    weights             = f["weights"][:]
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

# ─────────────────────────────────────────────
# Lumi-normalized weight
# ─────────────────────────────────────────────
weight_lumi_norm = weights * (XSEC * LUMI) / NGEN

# ─────────────────────────────────────────────
# Event index + best-perm charge
# ─────────────────────────────────────────────
Nevt        = best_perm.shape[0]
idx         = np.arange(Nevt)
charge_best = charge[idx, best_perm]

print(f"Total events      : {Nevt}")

# ─────────────────────────────────────────────
# pgof mask — identical to unrolled histograms
# pgof = exp(-0.5 * chi2) > 0.2  AND  chi2 > 0
# ─────────────────────────────────────────────
pgof = np.exp(-0.5 * best_chi2)
mask = (pgof > 0.2) & (best_chi2 > 0)
print(f"After pgof mask   : {np.sum(mask)}")

# ─────────────────────────────────────────────
# RECO: assign top/antitop from lepton charge
#   charge > 0 → leptonic side = top (t), hadronic = antitop (tbar)
#   charge < 0 → leptonic side = antitop (tbar), hadronic = top (t)
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# GEN: reconstruct Cartesian 4-momenta from
#   stored cylindrical components (pt, cosphi,
#   sinphi, sinheta=sinh(eta)) per best_perm
# ─────────────────────────────────────────────
gen_top_px_best     = gen_top_pt[idx, best_perm] * gen_top_cosphi[idx, best_perm]
gen_top_py_best     = gen_top_pt[idx, best_perm] * gen_top_sinphi[idx, best_perm]
gen_top_pz_best     = gen_top_pt[idx, best_perm] * gen_top_sinheta[idx, best_perm]
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

# ─────────────────────────────────────────────
# Apply pgof mask to all arrays
# ─────────────────────────────────────────────
mtt_reco       = mtt_reco_full[mask]
y_top_reco     = y_top_reco_full[mask]
y_antitop_reco = y_antitop_reco_full[mask]

mtt_gen        = mtt_gen_full[mask]
y_top_gen      = y_top_gen_full[mask]
y_antitop_gen  = y_antitop_gen_full[mask]

w = weight_lumi_norm[mask]
N = len(w)
print(f"Events after mask : {N}")

# ─────────────────────────────────────────────
# Mutually exclusive N+/N- conditions
#   N+: |y_t|    > Y0  AND  |y_tbar| <= Y0
#   N-: |y_tbar| > Y0  AND  |y_t|    <= Y0
#   both or neither → bin = -1 (excluded)
# Applied independently at reco and gen level
# ─────────────────────────────────────────────
# RECO
top_fwd_reco     = np.abs(y_top_reco)     > Y_0
antitop_fwd_reco = np.abs(y_antitop_reco) > Y_0
Nplus_reco       = top_fwd_reco  & ~antitop_fwd_reco
Nminus_reco      = antitop_fwd_reco & ~top_fwd_reco

# GEN
top_fwd_gen      = np.abs(y_top_gen)      > Y_0
antitop_fwd_gen  = np.abs(y_antitop_gen)  > Y_0
Nplus_gen        = top_fwd_gen   & ~antitop_fwd_gen
Nminus_gen       = antitop_fwd_gen  & ~top_fwd_gen

# ─────────────────────────────────────────────
# Get unrolled bin index per event
# -1 means event does not belong to any bin
# ─────────────────────────────────────────────
reco_bin = get_unrolled_bin(mtt_reco, Nplus_reco, Nminus_reco, reco_mtt_edges)
gen_bin  = get_unrolled_bin(mtt_gen,  Nplus_gen,  Nminus_gen,  gen_mtt_edges)

# ─────────────────────────────────────────────
# Categorise every event — mutually exclusive
#   in_matrix : valid reco AND valid gen → fills response matrix
#   is_fake   : valid reco, no valid gen → fills h_fakes
#   is_miss   : valid gen,  no valid reco → fills h_misses
#   is_lost   : neither valid → not filled anywhere
# ─────────────────────────────────────────────
has_reco  = reco_bin >= 0
has_gen   = gen_bin  >= 0

in_matrix = has_reco & has_gen
is_fake   = has_reco & ~has_gen
is_miss   = has_gen  & ~has_reco
is_lost   = ~has_reco & ~has_gen

# Verify partition is complete — no event double counted or lost
n_total_check = np.sum(in_matrix) + np.sum(is_fake) + np.sum(is_miss) + np.sum(is_lost)
assert n_total_check == N, \
    f"ERROR: categorisation sums to {n_total_check}, expected {N}"

print(f"\nEvent categorisation (after pgof mask):")
print(f"  In response matrix : {np.sum(in_matrix):>8d}  ({100*np.sum(in_matrix)/N:.1f}%)")
print(f"  Fakes              : {np.sum(is_fake):>8d}  ({100*np.sum(is_fake)/N:.1f}%)")
print(f"  Misses             : {np.sum(is_miss):>8d}  ({100*np.sum(is_miss)/N:.1f}%)")
print(f"  Lost               : {np.sum(is_lost):>8d}  ({100*np.sum(is_lost)/N:.1f}%)")
print(f"  Total              : {N:>8d}")
print(f"  Partition check    : PASSED")

# ─────────────────────────────────────────────
# Build axis labels
# ─────────────────────────────────────────────
reco_labels = make_labels(reco_mtt_edges)
gen_labels  = make_labels(gen_mtt_edges)

# ─────────────────────────────────────────────
# Response matrix TH2D
# Convention (matches TUnfold kHistMapOutputHoriz):
#   X axis = gen  (output, 8 bins)
#   Y axis = reco (input, 16 bins)
# Fill: x = gen bin centre, y = reco bin centre
# ─────────────────────────────────────────────
h_matrix = ROOT.TH2D("response_matrix",
                      "Response Matrix;Gen bin;Reco bin",
                      n_gen_bins,  0, n_gen_bins,
                      n_reco_bins, 0, n_reco_bins)
h_matrix.Sumw2()

for i, lbl in enumerate(gen_labels):
    h_matrix.GetXaxis().SetBinLabel(i + 1, lbl)
for i, lbl in enumerate(reco_labels):
    h_matrix.GetYaxis().SetBinLabel(i + 1, lbl)

# vectorised fill using numpy
g_arr = gen_bin[in_matrix].astype(float)  + 0.5
r_arr = reco_bin[in_matrix].astype(float) + 0.5
w_arr = w[in_matrix]
for g_val, r_val, w_val in zip(g_arr, r_arr, w_arr):
    h_matrix.Fill(g_val, r_val, w_val)

print(f"\nResponse matrix filled : {np.sum(in_matrix)} events")

# ─────────────────────────────────────────────
# Fakes TH1D (reco bins, no gen match)
# Interpretation: signal MC events that pass reco
# selection but have no valid gen-level counterpart.
# TUnfold uses this to correct for signal MC
# contamination outside the defined gen phase space.
# ─────────────────────────────────────────────
h_fakes = ROOT.TH1D("h_fakes",
                     "Fakes;Reco bin;Events",
                     n_reco_bins, 0, n_reco_bins)
h_fakes.Sumw2()
for i, lbl in enumerate(reco_labels):
    h_fakes.GetXaxis().SetBinLabel(i + 1, lbl)

r_fake = reco_bin[is_fake].astype(float) + 0.5
w_fake = w[is_fake]
for r_val, w_val in zip(r_fake, w_fake):
    h_fakes.Fill(r_val, w_val)

print(f"Fakes filled           : {np.sum(is_fake)} events")

# ─────────────────────────────────────────────
# Misses TH1D (gen bins, no reco match)
# Interpretation: gen-level events that were not
# reconstructed — encodes the selection efficiency.
# TUnfold uses this to correct for efficiency loss.
# ─────────────────────────────────────────────
h_misses = ROOT.TH1D("h_misses",
                      "Misses;Gen bin;Events",
                      n_gen_bins, 0, n_gen_bins)
h_misses.Sumw2()
for i, lbl in enumerate(gen_labels):
    h_misses.GetXaxis().SetBinLabel(i + 1, lbl)

g_miss = gen_bin[is_miss].astype(float) + 0.5
w_miss = w[is_miss]
for g_val, w_val in zip(g_miss, w_miss):
    h_misses.Fill(g_val, w_val)

print(f"Misses filled          : {np.sum(is_miss)} events")

# ─────────────────────────────────────────────
# Sanity checks
# ─────────────────────────────────────────────
print(f"\n{'='*60}")
print("Sanity checks")
print(f"{'='*60}")

tol = 1e-3  # absolute tolerance for floating point comparisons

# Check 1: for each reco bin, matrix row sum + fakes = reco total
print(f"\nCheck 1: matrix row sum + fakes = reco histogram content")
print(f"  {'Reco bin':<28} {'row sum':>10} {'fakes':>10} {'total':>10} {'ok?':>5}")
print(f"  {'-'*65}")
check1_ok = True
for r in range(n_reco_bins):
    row_sum  = sum(h_matrix.GetBinContent(g + 1, r + 1) for g in range(n_gen_bins))
    fake_val = h_fakes.GetBinContent(r + 1)
    total    = row_sum + fake_val
    # compute expected from numpy directly (avoids ROOT rounding)
    expected = np.sum(w[in_matrix][reco_bin[in_matrix] == r]) + np.sum(w[is_fake][reco_bin[is_fake] == r])
    ok = abs(total - expected) < tol
    if not ok:
        check1_ok = False
    print(f"  {reco_labels[r]:<28} {row_sum:>10.4f} {fake_val:>10.4f} {total:>10.4f} {'✓' if ok else '✗':>5}")
print(f"  Check 1: {'PASSED ✓' if check1_ok else 'FAILED ✗'}")

# Check 2: for each gen bin, matrix col sum + misses = gen total
print(f"\nCheck 2: matrix col sum + misses = gen histogram content")
print(f"  {'Gen bin':<28} {'col sum':>10} {'misses':>10} {'total':>10} {'ok?':>5}")
print(f"  {'-'*65}")
check2_ok = True
for g in range(n_gen_bins):
    col_sum  = sum(h_matrix.GetBinContent(g + 1, r + 1) for r in range(n_reco_bins))
    miss_val = h_misses.GetBinContent(g + 1)
    total    = col_sum + miss_val
    expected = np.sum(w[in_matrix][gen_bin[in_matrix] == g]) + np.sum(w[is_miss][gen_bin[is_miss] == g])
    ok = abs(total - expected) < tol
    if not ok:
        check2_ok = False
    print(f"  {gen_labels[g]:<28} {col_sum:>10.4f} {miss_val:>10.4f} {total:>10.4f} {'✓' if ok else '✗':>5}")
print(f"  Check 2: {'PASSED ✓' if check2_ok else 'FAILED ✗'}")

# Check 3: global weighted event budget
print(f"\nCheck 3: global weighted event budget")
total_matrix = sum(h_matrix.GetBinContent(g+1, r+1)
                   for g in range(n_gen_bins) for r in range(n_reco_bins))
total_fakes  = sum(h_fakes.GetBinContent(r+1)  for r in range(n_reco_bins))
total_misses = sum(h_misses.GetBinContent(g+1) for g in range(n_gen_bins))
total_lost   = np.sum(w[is_lost])
total_all    = np.sum(w)
total_used   = total_matrix + total_fakes + total_misses + total_lost

print(f"  Response matrix  : {total_matrix:>12.4f}")
print(f"  Fakes            : {total_fakes:>12.4f}")
print(f"  Misses           : {total_misses:>12.4f}")
print(f"  Lost (neither)   : {total_lost:>12.4f}")
print(f"  Sum of all       : {total_used:>12.4f}")
print(f"  Total weighted   : {total_all:>12.4f}")
print(f"  Difference       : {abs(total_used - total_all):>12.4e}")
check3_ok = abs(total_used - total_all) < 1e-6 * total_all
print(f"  Check 3: {'PASSED ✓' if check3_ok else 'FAILED ✗'}")

all_passed = check1_ok and check2_ok and check3_ok
print(f"\n{'='*60}")
print(f"Overall: {'ALL CHECKS PASSED ✓' if all_passed else 'SOME CHECKS FAILED ✗ — do not proceed to TUnfold'}")
print(f"{'='*60}")

# ─────────────────────────────────────────────
# Draw response matrix canvas
# Colour scale: zero bins shown as blue (minimum
# colour) by setting palette minimum to -epsilon
# so that exactly-zero bins are above the minimum
# and get the lowest palette colour, not white.
# ─────────────────────────────────────────────
c_matrix = ROOT.TCanvas("c_response_matrix", "Response Matrix", 1000, 800)
c_matrix.SetLeftMargin(0.22)
c_matrix.SetBottomMargin(0.22)
c_matrix.SetRightMargin(0.14)
c_matrix.SetTopMargin(0.08)

ROOT.gStyle.SetPalette(ROOT.kBird)

h_matrix.GetXaxis().SetLabelSize(0.028)
h_matrix.GetYaxis().SetLabelSize(0.028)
h_matrix.GetXaxis().LabelsOption("v")
h_matrix.GetZaxis().SetTitle("Weighted events")
h_matrix.GetZaxis().SetTitleOffset(1.2)

# set z range: min slightly below zero so zero bins
# get the bottom palette colour (blue) not white
z_max = h_matrix.GetMaximum()
h_matrix.SetMinimum(-1e-6 * z_max)
h_matrix.SetMaximum(z_max)

h_matrix.Draw("COLZ")
c_matrix.Update()

# force palette range after first draw
h_matrix.GetZaxis().SetRangeUser(-1e-6 * z_max, z_max)
c_matrix.Modified()
c_matrix.Update()

c_matrix.SaveAs(f"{OUTDIR}/response_matrix.pdf")
c_matrix.SaveAs(f"{OUTDIR}/response_matrix.png")
print(f"\nSaved: {OUTDIR}/response_matrix.pdf / .png")

# ─────────────────────────────────────────────
# Write to ROOT file
# ─────────────────────────────────────────────
fout = ROOT.TFile(OUT_FILE, "RECREATE")
h_matrix.Write()    # TH2D — response matrix for TUnfold
h_fakes.Write()     # TH1D — fakes for TUnfold
h_misses.Write()    # TH1D — misses for TUnfold
c_matrix.Write("canvas_response_matrix")  # styled TCanvas for display
fout.Close()

print(f"\n{'='*60}")
print(f"Done. Output: {OUT_FILE}")
print(f"  response_matrix        → TH2D ({n_gen_bins} gen x {n_reco_bins} reco bins)")
print(f"  h_fakes                → TH1D ({n_reco_bins} reco bins)")
print(f"  h_misses               → TH1D ({n_gen_bins} gen bins)")
print(f"  canvas_response_matrix → styled TCanvas")
