import os
import json
import pandas as pd

def load_data() -> pd.DataFrame:
    """
    This function loads all the CSV files from the DATA_DIR into DataFrames
    which are then concatenated into a single DataFrame that is returned.
    """
    with open(os.path.join("config", "data_config.json"), "rb") as f:
        data_config = json.load(f)
        DATA_DIR = data_config["DATA_DIR"]
        COL_DTYPES = data_config["COL_DTYPES"]

    df = pd.DataFrame()

    for fn in os.listdir(DATA_DIR):
        if fn.endswith(".csv"):
            df_ = pd.read_csv(os.path.join(DATA_DIR, fn), dtype=COL_DTYPES)
            df = pd.concat([df, df_])

    return df