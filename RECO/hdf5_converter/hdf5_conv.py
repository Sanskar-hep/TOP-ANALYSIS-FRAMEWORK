import awkward as ak
import numpy as np
import h5py
import glob
import os

# -----------------------------
# Input / Output configuration
# -----------------------------
input_pattern = "/mnt/disk2/sanskar/2016preVFP_RECO/RECO_UPDATED_MUKUND/*.parquet"
output_dir = "./h5_output"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get list of parquet files
parquet_files = glob.glob(input_pattern)

print(f"Found {len(parquet_files)} parquet files")

# -----------------------------
# Processing loop
# -----------------------------
for pq_file in parquet_files:
    try:
        print(f"\nProcessing: {pq_file}")

        # Load parquet file
        data = ak.from_parquet(pq_file)

        # Convert to numpy
        numpy_arrays = {}
        for field in data.fields:
            numpy_arrays[field] = ak.to_numpy(data[field])

        # Construct output filename
        base_name = os.path.basename(pq_file).replace(".parquet", ".h5")
        output_path = os.path.join(output_dir, base_name)

        print(f"Saving to: {output_path}")

        # Write to HDF5
        with h5py.File(output_path, "w") as f:
            for name, arr in numpy_arrays.items():
                f.create_dataset(name, data=arr, compression="gzip")

        print("Done ✔")

    except Exception as e:
        print(f"Failed for {pq_file}")
        print(f"Error: {e}")
