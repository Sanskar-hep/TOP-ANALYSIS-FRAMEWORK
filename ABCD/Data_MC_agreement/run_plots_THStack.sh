#!/bin/bash
# ==============================================================
# run_plots.sh — run the stacked histogram macro for a given era
#
# Usage:
#   ./run_plots.sh 2018
#   ./run_plots.sh 2017
#   ./run_plots.sh 2016preVFP
#   ./run_plots.sh 2016postVFP
# ==============================================================

ERA=${1}   # first argument

# ── Validate era ──────────────────────────────────────────────
VALID_ERAS=("2016preVFP" "2016postVFP" "2017" "2018")
VALID=false
for e in "${VALID_ERAS[@]}"; do
    if [[ "$ERA" == "$e" ]]; then
        VALID=true
        break
    fi
done

if [[ -z "$ERA" ]]; then
    echo "ERROR: no era specified."
    echo "Usage: ./run_plots.sh <era>"
    echo "       era options: 2016preVFP  2016postVFP  2017  2018"
    exit 1
fi

if [[ "$VALID" == false ]]; then
    echo "ERROR: unknown era '$ERA'"
    echo "Valid options: 2016preVFP  2016postVFP  2017  2018"
    exit 1
fi

echo "=============================================="
echo "  Running plots for era: $ERA"
echo "=============================================="

root -l -b -q "plot_stacked_histograms_thesis.C(\"${ERA}\")"

STATUS=$?
if [[ $STATUS -eq 0 ]]; then
    echo ""
    echo "Done. Plots saved to plots/*_stack_${ERA}.png/.pdf"
    echo "ROOT file: plots/all_plots_${ERA}.root"
else
    echo "ERROR: ROOT macro failed with exit code $STATUS"
    exit $STATUS
fi
