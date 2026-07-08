#!/usr/bin/env python3
"""
Coffea → ROOT converter with proper Sumw2 (weighted error) preservation.

Coffea file structure assumed:
    output[dataset][dataset][hist_name]  →  hist.Hist object

ROOT output structure:
    hist_name/dataset_clean  →  TH1D with Sumw2 filled
    (grouped by histogram name so stack plots are easy to loop over)
"""

from coffea.util import load
import uproot
import numpy as np
import os
import sys


class CoffeaToROOTConverter:

    def __init__(self, verbose=True):
        self.verbose = verbose

    def log(self, msg):
        if self.verbose:
            print(msg)

    # ------------------------------------------------------------------
    # Unwrap {dataset: {dataset: {hist_name: hist}}} safely
    # ------------------------------------------------------------------
    def get_inner(self, output, dataset):
        """
        Navigate the double-key nesting coffea sometimes produces.
        Returns the dict of {hist_name: hist} or None on failure.
        """
        level1 = output[dataset]

        # Expected: level1 is a dict whose first (or same-name) key holds hists
        if not isinstance(level1, dict):
            self.log(f"  WARNING: {dataset} value is not a dict, skipping.")
            return None

        # Prefer the same-name inner key, else fall back to first key
        if dataset in level1:
            inner = level1[dataset]
        else:
            keys = list(level1.keys())
            if not keys:
                self.log(f"  WARNING: {dataset} inner dict is empty, skipping.")
                return None
            inner = level1[keys[0]]
            self.log(f"  INFO: inner key '{keys[0]}' used for dataset '{dataset}'")

        return inner

    # ------------------------------------------------------------------
    # Extract values, Sumw2, edges from a hist.Hist object
    # ------------------------------------------------------------------
    def extract_1d(self, h, hist_name):
        """
        Returns (values, sumw2, edges) for a 1-D weighted histogram,
        or None if extraction fails.

        * values  – sum of weights per bin  (what goes into bin content)
        * sumw2   – sum of weights^2 per bin (what goes into TH1::Sumw2)
        * edges   – bin edges array, length = len(values)+1
        """
        # ---- dimensionality check ----
        if hasattr(h, 'ndim') and h.ndim != 1:
            self.log(f"    SKIP {hist_name}: {h.ndim}-D histogram (only 1-D supported)")
            return None

        try:
            # --- bin contents (sum of weights) ---
            values = np.asarray(h.values(), dtype=float)

            # --- sum of weights^2 ---
            # h.variances() returns Sumw2 for weighted fills; fall back to |values|
            if h.variances() is not None:
                sumw2 = np.asarray(h.variances(), dtype=float)
                # variances() can contain tiny negatives due to float arithmetic
                sumw2 = np.maximum(sumw2, 0.0)
            else:
                self.log(f"    WARNING {hist_name}: no variance storage, using |values| as Sumw2")
                sumw2 = np.abs(values)

            # --- bin edges ---
            edges = np.asarray(h.axes[0].edges, dtype=float)

        except Exception as exc:
            self.log(f"    ERROR extracting {hist_name}: {exc}")
            return None

        # ---- sanity checks ----
        if len(values) + 1 != len(edges):
            self.log(f"    ERROR {hist_name}: #values={len(values)} but #edges={len(edges)}")
            return None
        if len(values) != len(sumw2):
            self.log(f"    ERROR {hist_name}: #values={len(values)} but #sumw2={len(sumw2)}")
            return None

        return values, sumw2, edges

    # ------------------------------------------------------------------
    # Write a TH1D with Sumw2 into an open uproot file
    # ------------------------------------------------------------------
    def write_th1d(self, root_file, path, values, sumw2, edges):
        """
        Writes a TH1D to *root_file* at *path*, preserving Sumw2.

        uproot ≥ 5 accepts a boost-histogram / hist object directly via
        `to_boost()`, which carries the storage and therefore Sumw2.
        We reconstruct a minimal boost-histogram here so we never depend
        on the original hist object being in scope.
        """
        import boost_histogram as bh

        # Build a boost-histogram with Weight storage (holds value + variance)
        axis = bh.axis.Variable(edges)
        bh_hist = bh.Histogram(axis, storage=bh.storage.Weight())

        # .view() returns a structured array with fields 'value' and 'variance'
        view = bh_hist.view()
        view["value"]    = values
        view["variance"] = sumw2   # variance == sumw2 for weighted fills

        try:
            root_file[path] = bh_hist          # uproot serialises Weight storage → Sumw2
        except Exception as exc:
            self.log(f"    ERROR writing {path}: {exc}")
            raise

    # ------------------------------------------------------------------
    # Main conversion entry point
    # ------------------------------------------------------------------
    def convert_file(self, coffea_path, root_path, histograms_to_convert=None):
        """
        Convert *coffea_path* → *root_path*.

        ROOT directory layout (good for stack plots):
            <hist_name>/<dataset_clean>   ← one TH1D per sample
        """
        self.log(f"\n{'='*65}")
        self.log(f"Input : {coffea_path}")
        self.log(f"Output: {root_path}")
        self.log(f"{'='*65}")

        if not os.path.exists(coffea_path):
            self.log(f"ERROR: file not found: {coffea_path}")
            return False

        try:
            self.log("Loading coffea file …")
            output = load(coffea_path)
            self.log(f"Loaded. Top-level datasets: {list(output.keys())}")
        except Exception as exc:
            self.log(f"FATAL: could not load coffea file: {exc}")
            return False

        out_dir = os.path.dirname(root_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)

        SKIP_KEYS = {"selEvents", "sumw", "cutflow"}   # non-histogram entries

        total = successful = 0

        with uproot.recreate(root_path) as rf:
            for dataset in output.keys():
                self.log(f"\n── Dataset: {dataset}")

                inner = self.get_inner(output, dataset)
                if inner is None:
                    continue

                # Clean name: ROOT directory names must not contain / : space
                ds_clean = dataset.replace("/", "_").replace(":", "_").replace(" ", "_")

                hist_names = list(inner.keys()) if hasattr(inner, "keys") else []
                self.log(f"   {len(hist_names)} histogram keys found")

                for hname in hist_names:
                    if hname in SKIP_KEYS:
                        self.log(f"   skip  {hname}  (non-histogram key)")
                        continue
                    if histograms_to_convert and hname not in histograms_to_convert:
                        continue

                    total += 1
                    try:
                        h = inner[hname]
                        result = self.extract_1d(h, hname)
                        if result is None:
                            continue

                        values, sumw2, edges = result
                        self.log(
                            f"   conv  {hname:30s}  "
                            f"bins={len(values)}  "
                            f"range=[{edges[0]:.2f}, {edges[-1]:.2f}]  "
                            f"integral={values.sum():.3g}"
                        )

                        # Layout: hist_name/dataset  → easy to THStack in ROOT macro
                        self.write_th1d(rf, f"{hname}/{ds_clean}", values, sumw2, edges)
                        successful += 1

                    except Exception as exc:
                        self.log(f"   ERROR {hname}: {exc}")
                        continue

        self.log(f"\n{'='*65}")
        self.log(f"Done. Converted {successful}/{total} histograms → {root_path}")
        self.log(f"{'='*65}\n")
        return successful > 0


# ──────────────────────────────────────────────────────────────────────
def main():
    converter = CoffeaToROOTConverter(verbose=True)

    histograms = [
        "electron_pt", "electron_eta", "electron_phi",
        "Jet_pt",      "Jet_eta",      "Jet_phi",
    ]

    files = [
        (
            "regionD_ABCD_for_2017_with_nbtags_vs_id.coffea",
            "regionD_ABCD_for_2017_with_nbtags_vs_id.root",
        ),
        # Uncomment to also convert QCD estimate:
        # (
        #     "QCD_ESTIMATE_ALL_HISTOGRAMS_fromB.coffea",
        #     "QCD_ESTIMATE_ALL_HISTOGRAMS_fromB.root",
        # ),
    ]

    ok = sum(converter.convert_file(c, r, histograms) for c, r in files)
    total = len(files)

    print(f"\n{'='*65}")
    print(f"SUMMARY: {ok}/{total} files converted successfully.")
    print(f"{'='*65}")

    if ok == total:
        print("\n✓ All done!")
        print("\nQuick verify in ROOT:")
        print("  root -l regionD_ABCD_for_2018_with_nbtags_vs_id.root")
        print('  _file0->ls();')
        print('  ((TH1D*)_file0->Get("electron_pt/<dataset>"))->Draw();')
        print("\nIn your stack macro, loop over datasets inside each hist_name directory.")
    else:
        print("\n✗ Some files failed – check messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
