import pandas as pd

from inference import run_inference

# User-configurable inputs
csvFileLoc = r"./data/ages.csv" # file containing chronological ages of the brains in the brainsDir
brainsDir = r"./data/"

# Load metadata if needed for downstream use
data_df = pd.read_csv(csvFileLoc)

saveFlag = True  # this flag controls whether to save the output predictions as .npy files
saveLoc = r"./outFiles/"

# Execute inference using explicit arguments
_ = data_df
run_inference(
    brains_dir=brainsDir,
    save_flag=saveFlag,
    save_loc=saveLoc,
)
