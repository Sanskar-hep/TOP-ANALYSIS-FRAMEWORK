#!/bin/bash

# ==============================================================================
# Script to run b-tag efficiency analysis for all regions and years
# ==============================================================================

# --- Configuration ---
YEAR="2016preVFP"  # Change this to run for different years
BTAG_WP="M"         # B-tagging working point: L, M, or T
PILEUP_WP="L"       # Pileup working point: L, M, or T
JET_PT=20.0
CHOICE=2
# --- List of regions to run ---
regions=("D")

echo "============================================================"
echo ">>> Starting ABCD analysis"
echo ">>> Year: $YEAR"
echo ">>> B-tag WP: $BTAG_WP"
echo ">>> PileUp WP: $PILEUP_WP"
echo ">>> Regions: ${regions[@]}"
echo "============================================================"
echo ""

# --- Run dataset.py once at the beginning ---
echo "--- Preparing dataset ---"
python3 dataset.py
if [ $? -ne 0 ]; then
    echo "Error: dataset.py failed. Exiting..."
    exit 1
fi
echo ""

# --- Loop through all regions ---
for region_code in "${regions[@]}"; do
    echo "============================================================"
    echo ">>> Starting analysis for Region $region_code"
    echo "============================================================"
    
    # --- Run the analysis ---
    python3 region_runner.py \
        --year "$YEAR" \
        --region "$region_code" \
        --btagWP "$BTAG_WP" \
        --pileUpWP "$PILEUP_WP" \
	--jetpt "$JET_PT" \
        --choice "$CHOICE"

    # --- Check if the run was successful ---
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Successfully completed Region $region_code"
        echo ""
    else
        echo ""
        echo "✗ Error in Region $region_code - Check logs above"
        echo ""
        # Uncomment the line below if you want to stop on first error
        # exit 1
    fi
done

echo "============================================================"
echo ">>> All regions finished for year $YEAR"
echo "============================================================"
