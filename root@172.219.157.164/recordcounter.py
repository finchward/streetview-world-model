import h5py

file_path = "dataset.h5"

with h5py.File(file_path, "r") as f:
    num_steps = f["images"].shape[0]  # number of records along the first dimension
    result = num_steps * 64
    print(f"Number of records: {num_steps}")
    print(f"Number of records multiplied by 64: {result}")
