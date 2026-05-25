#!/bin/bash

INPUT_DIR="."
SCRIPT="chi2_optimize_new.py"
PROCESSED_LOG="/home/irfan/sanskar/outputs/2016preVFP_reco/RECO_REVISIT/processed_files.txt"

# Create log file if not exists
touch "$PROCESSED_LOG"

while true; do
    echo "========================================"
    echo "Available input files:"
    echo "========================================"

    # Only pick your reco files
    files=( *_reco_variables_updated.h5 )

    # Check if no files found
    if [ ${#files[@]} -eq 0 ]; then
        echo "No input files found!"
        exit 1
    fi

    # Display with status
    for i in "${!files[@]}"; do
        fname="${files[$i]}"

        if grep -Fxq "$fname" "$PROCESSED_LOG"; then
            status="[✔]"
        else
            status="[ ]"
        fi

        echo "$((i+1))) $status $fname"
    done

    echo "----------------------------------------"
    echo "Enter file number to process (or 'q' to quit):"
    read choice

    # Exit condition
    if [[ "$choice" == "q" ]]; then
        echo "Exiting."
        break
    fi

    # Validate input
    if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt "${#files[@]}" ]; then
        echo "Invalid selection. Try again."
        continue
    fi

    selected_file="${files[$((choice-1))]}"

    # Check if already processed
    if grep -Fxq "$selected_file" "$PROCESSED_LOG"; then
        echo "WARNING: '$selected_file' already processed!"
        continue
    fi

    echo "----------------------------------------"
    echo "Running: $SCRIPT on $selected_file"
    echo "----------------------------------------"

    # Run (blocking)
    python "$SCRIPT" "$selected_file"

    # Check exit status
    if [ $? -eq 0 ]; then
        echo "$selected_file" >> "$PROCESSED_LOG"
        echo "Completed: $selected_file"
    else
        echo "Failed: $selected_file"
    fi

    echo "----------------------------------------"
    echo "Done. Select next file."
    echo "----------------------------------------"

done
