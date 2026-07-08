#!/usr/bin/env python3
"""
Fixed QCD converter - with proper Sumw2 (weighted error) preservation.

Coffea structure:  qcd_data['QCD'][hist_name]   (single-level, no double nesting)

ROOT output layout:  hist_name/QCD
  e.g.  electron_pt/QCD
        electron_eta/QCD
        Jet_pt/QCD
        Jet_eta/QCD

This matches the macro's LoadHist(qcd_file, hist_name, "QCD") call.
"""
from coffea.util import load
import uproot
import numpy as np
import boost_histogram as bh


def convert_qcd_file():
    print("=" * 70)
    print("Converting QCD File to ROOT Format")
    print("=" * 70)

    print("\nLoading QCD_ESTIMATE_ALL_HISTOGRAMS_fromB.coffea...")
    qcd_data = load("QCD_ESTIMATE_ALL_HISTOGRAMS_fromB.coffea")
    print("✓ File loaded successfully")

    # Structure is flat: qcd_data['QCD'][hist_name]
    qcd_histograms = qcd_data['QCD']

    print(f"\nFound {len(qcd_histograms)} histograms:")
    for name in qcd_histograms.keys():
        print(f"  - {name}")

    print("\n" + "=" * 70)
    print("Converting to ROOT format...")
    print("=" * 70)

    SKIP_KEYS = {'selEvents', 'sumw', 'cutflow'}

    with uproot.recreate("QCD_ESTIMATE_ALL_HISTOGRAMS_fromB.root") as root_file:

        converted_count = 0

        for hist_name, h in qcd_histograms.items():

            if hist_name in SKIP_KEYS:
                print(f"\n  Skipping: {hist_name} (not a histogram)")
                continue

            try:
                # Extract data using public API
                values = np.asarray(h.values(),       dtype=float)
                edges  = np.asarray(h.axes[0].edges,  dtype=float)

                raw_var = h.variances()
                if raw_var is not None:
                    sumw2 = np.maximum(np.asarray(raw_var, dtype=float), 0.0)
                else:
                    print(f"    WARNING: no variance storage, using |values| as Sumw2")
                    sumw2 = np.abs(values)

                print(f"\n  {hist_name}:")
                print(f"    Bins    : {len(values)}")
                print(f"    Range   : [{edges[0]:.3f}, {edges[-1]:.3f}]")
                print(f"    Integral: {values.sum():.4f}")
                print(f"    Sumw2   : {sumw2.sum():.4f}  (non-zero = errors preserved)")

                # Build boost-histogram with Weight storage
                axis    = bh.axis.Variable(edges)
                bh_hist = bh.Histogram(axis, storage=bh.storage.Weight())
                view = bh_hist.view()
                view["value"]    = values
                view["variance"] = sumw2

                # Write with layout: hist_name/QCD
                # This matches the ROOT macro's LoadHist(qcd_file, hist_name, "QCD")
                path = f"{hist_name}/QCD"           # was "QCD/{hist_name}" before
                root_file[path] = bh_hist
                print(f"    Written to: {path}")

                converted_count += 1

            except Exception as e:
                print(f"    Error converting {hist_name}: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 70)
    print("CONVERSION COMPLETE!")
    print("=" * 70)
    print(f"\nSuccessfully converted {converted_count} QCD histograms")
    print("Output file: QCD_ESTIMATE_ALL_HISTOGRAMS_fromB.root")

    print("\nVerify the ROOT file layout:")
    print("  python3 -c \"import uproot; f=uproot.open('QCD_ESTIMATE_ALL_HISTOGRAMS_fromB.root'); print(f.keys())\"")
    print("  # Expected: ['electron_pt/QCD;1', 'electron_eta/QCD;1', ...]")

    return True


if __name__ == "__main__":
    try:
        convert_qcd_file()
        print("\nSuccess! Now re-run the ROOT plotting macro:")
        print("   root -l -b -q 'plot_stacked_histograms.C'")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
