import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the Breast Cancer dataset from Excel or CSV.
    Returns a pandas DataFrame.
    """
    if file_path.lower().endswith((".xls", ".xlsx")):
        df = pd.read_excel(file_path)
    elif file_path.lower().endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file type for: {file_path}")
    return df
