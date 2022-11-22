import pandas as pd

def load_to_df_from_csv(csv_file_path):
    df = pd.read_csv(csv_file_path)
    return df