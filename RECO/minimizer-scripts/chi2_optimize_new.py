"""
================================================================================
Kinematic Fit for ttbar Semileptonic Events
================================================================================

Description:
    This script performs a kinematic fit on ttbar semileptonic decay events.
    It minimizes a chi-squared function to find the best assignment of
    reconstructed objects (electron, neutrino, b-jets, light jets) to the
    decay products of the top and antitop quarks.

Decay Topology:
    ttbar -> (W+ -> e+ nu)(W- -> j1 j2)(bl)(bh)
    
    Leptonic side : t  -> W_lep (-> e + nu)  + b_lep
    Hadronic side : t  -> W_had (-> j1 + j2) + b_had

Chi-squared Definition:
    chi2 = sum((x - x_meas)^2 / res^2)          [measurement term]
         + ((mWl - MW)  / sigmaW)^2              [leptonic W mass]
         + ((mWh - MW)  / sigmaW)^2              [hadronic W mass]
         + ((mtl - mth) / sigmaT)^2              [top mass equality]

    where:
        MW     = 80.4  GeV  (W boson mass)
        MT     = 172.5 GeV  (top quark mass)
        sigmaW = 10.0  GeV  (W mass resolution)
        sigmaT = 13.0  GeV  (top mass resolution)

Permutations:
    4 permutations corresponding to different b-jet and light jet assignments.
    The permutation with the lowest chi2 is selected as the best.

Optimizer:
    scipy.optimize.minimize with method = SLSQP

Parallelization:
    Python multiprocessing.Pool with imap_unordered for event-level parallelism.
    Number of processes = min(cpu_count, 12)

Author  : [Sanskar Nanda]
Date    : [8th March, 2026]
Version : 1.0
================================================================================
"""

import numpy as np
import h5py
from scipy.optimize import minimize
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import sys
import os

# ============================================================
# 1. Physics constants
# ============================================================

MW = 80.4
MT = 172.5
sigmaW = 10.0
sigmaT = 13.0

# ============================================================
# 2. Kinematic utilities
# ============================================================

def energy(px, py, pz, m):
    return np.sqrt(px**2 + py**2 + pz**2 + m**2)

def inv_mass(px, py, pz, E):
    m2 = E**2 - (px**2 + py**2 + pz**2)
    return np.sqrt(max(m2, 0.0))

# ============================================================
# 3. Unpack 18-component momentum vector
# ============================================================

def unpack_x(x, masses):
    e_px, e_py, e_pz = x[0:3]
    nu_px, nu_py, nu_pz = x[3:6]
    bl_px, bl_py, bl_pz = x[6:9]
    bh_px, bh_py, bh_pz = x[9:12]
    j1_px, j1_py, j1_pz = x[12:15]
    j2_px, j2_py, j2_pz = x[15:18]

    return {
        "e":  (e_px, e_py, e_pz, energy(e_px, e_py, e_pz, 0.0)),
        "nu": (nu_px, nu_py, nu_pz, energy(nu_px, nu_py, nu_pz, 0.0)),
        "bl": (bl_px, bl_py, bl_pz, energy(bl_px, bl_py, bl_pz, masses["b_lep"])),
        "bh": (bh_px, bh_py, bh_pz, energy(bh_px, bh_py, bh_pz, masses["b_had"])),
        "j1": (j1_px, j1_py, j1_pz, energy(j1_px, j1_py, j1_pz, masses["l1"])),
        "j2": (j2_px, j2_py, j2_pz, energy(j2_px, j2_py, j2_pz, masses["l2"]))
    }

# ============================================================
# 4. χ² definition
# ============================================================

def chi2_function(x, x_m, res_safe, masses):
    #res_safe = np.where(res==0, 1e-6, res)
    chi2_meas = np.sum(((x - x_m) / res_safe)**2)


    #print(np.sum(res==0))
    obj = unpack_x(x, masses)

    # leptonic W
    Wl_px = obj["e"][0] + obj["nu"][0]
    Wl_py = obj["e"][1] + obj["nu"][1]
    Wl_pz = obj["e"][2] + obj["nu"][2]
    Wl_E  = obj["e"][3] + obj["nu"][3]
    mWl = inv_mass(Wl_px, Wl_py, Wl_pz, Wl_E)

    # hadronic W
    Wh_px = obj["j1"][0] + obj["j2"][0]
    Wh_py = obj["j1"][1] + obj["j2"][1]
    Wh_pz = obj["j1"][2] + obj["j2"][2]
    Wh_E  = obj["j1"][3] + obj["j2"][3]
    mWh = inv_mass(Wh_px, Wh_py, Wh_pz, Wh_E)

    # leptonic top
    tl_px = Wl_px + obj["bl"][0]
    tl_py = Wl_py + obj["bl"][1]
    tl_pz = Wl_pz + obj["bl"][2]
    tl_E  = Wl_E  + obj["bl"][3]
    mtl = inv_mass(tl_px, tl_py, tl_pz, tl_E)

    # hadronic top
    th_px = Wh_px + obj["bh"][0]
    th_py = Wh_py + obj["bh"][1]
    th_pz = Wh_pz + obj["bh"][2]
    th_E  = Wh_E  + obj["bh"][3]
    mth = inv_mass(th_px, th_py, th_pz, th_E)

    chi2_mass = (
        ((mWl - MW) / sigmaW)**2 +
        ((mWh - MW) / sigmaW)**2 +
        ((mtl - mth) / sigmaT)**2
    )

    return chi2_meas + chi2_mass

# ============================================================
# 5. Fit one permutation
# ============================================================

def fit_one_perm(x_m_p, res_p, masses):

    res_safe = np.where(res_p==0, 1e-6, res_p)
    result = minimize(
        chi2_function,
        x_m_p,
        args=(x_m_p, res_safe, masses),
        method="SLSQP",
        options = {"maxiter":1000,"ftol":1e-6}
    )

    if not result.success:
        return np.inf, x_m_p
    
    return result.fun, result.x

# ============================================================
# 6. Fit one event (4 permutations)
# ============================================================

def fit_event(evt):
    chi2_vals = np.zeros(4)
    x_fit = np.zeros((4, 18))

    for p in range(4):
        masses = {
            "b_lep": b_lep_mass[evt, p],
            "b_had": b_had_mass[evt, p],
            "l1":    l1_mass[evt, p],
            "l2":    l2_mass[evt, p],
        }

        chi2_vals[p], x_fit[p] = fit_one_perm(
            x_m[evt, p],
            res[evt, p],
            masses
        )

    best = np.argmin(chi2_vals)

    if not np.isfinite(chi2_vals[best]):
        return best, -1.0, x_m[evt, best]
    
    return best, chi2_vals[best], x_fit[best]

# ============================================================
# 7. Reconstructed top masses
# ============================================================

def reco_top_masses(x_fit, masses):
    obj = unpack_x(x_fit, masses)

    Wl_px = obj["e"][0] + obj["nu"][0]
    Wl_py = obj["e"][1] + obj["nu"][1]
    Wl_pz = obj["e"][2] + obj["nu"][2]
    Wl_E  = obj["e"][3] + obj["nu"][3]

    tl_px = Wl_px + obj["bl"][0]
    tl_py = Wl_py + obj["bl"][1]
    tl_pz = Wl_pz + obj["bl"][2]
    tl_E  = Wl_E  + obj["bl"][3]

    Wh_px = obj["j1"][0] + obj["j2"][0]
    Wh_py = obj["j1"][1] + obj["j2"][1]
    Wh_pz = obj["j1"][2] + obj["j2"][2]
    Wh_E  = obj["j1"][3] + obj["j2"][3]

    th_px = Wh_px + obj["bh"][0]
    th_py = Wh_py + obj["bh"][1]
    th_pz = Wh_pz + obj["bh"][2]
    th_E  = Wh_E  + obj["bh"][3]

    return (
        tl_px, tl_py, tl_pz, tl_E,
        th_px, th_py, th_pz, th_E
    )

# ============================================================
# 8. Multiprocessing wrapper (ONE EVENT)
# ============================================================

def process_one_event(i):
    bp, chi2, xfit = fit_event(i)

    masses = {
        "b_lep": b_lep_mass[i, bp],
        "b_had": b_had_mass[i, bp],
        "l1":    l1_mass[i, bp],
        "l2":    l2_mass[i, bp],
    }
    
    tl_px, tl_py, tl_pz, tl_E,th_px, th_py, th_pz,th_E = reco_top_masses(xfit, masses)
    return i, bp, chi2, xfit, tl_px, tl_py, tl_pz, tl_E, th_px, th_py, th_pz, th_E

# ============================================================
# 9. MAIN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv)!=2:
        print("Usage: python minimizer.py <input.h5>")
        sys.exit(1)

    input_file = sys.argv[1]
    basename = os.path.splitext(os.path.basename(input_file))[0]

    with h5py.File(input_file, "r") as f:
        x_m        = f["x_m"][:]
        res        = f["res"][:]
        b_lep_mass = f["b_lep_mass"][:]
        b_had_mass = f["b_had_mass"][:]
        l1_mass    = f["l1_mass"][:]
        l2_mass    = f["l2_mass"][:]

    
    Nevt = x_m.shape[0]
    #Nevt = 5000
    print(f"Loaded {Nevt} events")

    best_perm = np.zeros(Nevt, dtype=int)
    best_chi2 = np.zeros(Nevt)
    x_fit_best = np.zeros((Nevt, 18))
    top_lep_px = np.zeros(Nevt)
    top_lep_py = np.zeros(Nevt)
    top_lep_pz = np.zeros(Nevt)
    top_lep_E = np.zeros(Nevt)
    top_had_px = np.zeros(Nevt)
    top_had_py = np.zeros(Nevt)
    top_had_pz = np.zeros(Nevt)
    top_had_E = np.zeros(Nevt)

    #nproc = min(cpu_count()-2, 17)
    nproc = cpu_count()
    print(f"The minimizer is using : {nproc} cores")
    
    results = []
    with Pool(processes=nproc) as pool:
        for res_ev in tqdm(
            pool.imap_unordered(process_one_event, range(Nevt)),
            total=Nevt,
            desc="Kinematic fit"
        ):
            results.append(res_ev)

    for i, bp, chi2, xfit, tl_px, tl_py, tl_pz, tl_E, th_px, th_py, th_pz, th_E in results:
        best_perm[i] = bp
        best_chi2[i] = chi2
        x_fit_best[i] = xfit
        top_lep_px[i] = tl_px
        top_lep_py[i] = tl_py
        top_lep_pz[i] = tl_pz
        top_lep_E[i] = tl_E
        top_had_px[i] = th_px
        top_had_py[i] = th_py
        top_had_pz[i] = th_pz
        top_had_E[i] = th_E

    
    #output_file = "/mnt/disk2/sanskar/2018Analysis/RECO/ttbar_semilep_fit_2018.h5"
    output_file = f"/home/irfan/sanskar/outputs/2016preVFP_reco/RECO_REVISIT/{basename}_fit_2016preVFP.h5"
    
    with h5py.File(output_file, "w") as f:
        f.create_dataset("best_perm", data=best_perm)
        f.create_dataset("best_chi2", data=best_chi2)
        f.create_dataset("x_fit", data=x_fit_best)
        f.create_dataset("top_lep_px", data=top_lep_px)
        f.create_dataset("top_lep_py",data=top_lep_py)
        f.create_dataset("top_lep_pz",data=top_lep_pz)
        f.create_dataset("top_lep_E",data=top_lep_E)
        f.create_dataset("top_had_px",data=top_had_px)
        f.create_dataset("top_had_py",data=top_had_py)
        f.create_dataset("top_had_pz",data=top_had_pz)
        f.create_dataset("top_had_E",data=top_had_E)

        
    print(" Kinematic fit completed with multiprocessing + progress bar.")
