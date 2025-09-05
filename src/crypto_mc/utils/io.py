
import os
import zipfile
import pandas as pd
from typing import Optional, List

def find_data_files(data_path: str) -> List[str]:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path not found: {data_path}")
    files = []
    if os.path.isdir(data_path):
        for name in os.listdir(data_path):
            if name.lower().endswith(".csv"):
                files.append(os.path.join(data_path, name))
            elif name.lower().endswith(".zip"):
                files.append(os.path.join(data_path, name))
    else:
        files.append(data_path)
    if not files:
        raise FileNotFoundError(f"No CSV/ZIP found in {data_path}")
    return files

def load_coin_csv_from_zip(zip_path: str, symbol: str) -> Optional[pd.DataFrame]:
    with zipfile.ZipFile(zip_path, 'r') as z:
        for member in z.namelist():
            if member.lower().endswith('.csv'):
                with z.open(member) as f:
                    df = pd.read_csv(f)
                if 'Symbol' in df.columns and df['Symbol'].iloc[0].upper() == symbol.upper():
                    df['source_file'] = os.path.basename(member)
                    return df
    return None

def load_coin_csv(path: str, symbol: str) -> Optional[pd.DataFrame]:
    import pandas as pd
    if path.lower().endswith('.zip'):
        return load_coin_csv_from_zip(path, symbol)
    else:
        if os.path.isdir(path):
            for fname in os.listdir(path):
                fpath = os.path.join(path, fname)
                if fpath.lower().endswith('.csv'):
                    df = pd.read_csv(fpath)
                    if 'Symbol' in df.columns and df['Symbol'].iloc[0].upper() == symbol.upper():
                        df['source_file'] = os.path.basename(fpath)
                        return df
        else:
            df = pd.read_csv(path)
            if 'Symbol' in df.columns and df['Symbol'].iloc[0].upper() == symbol.upper():
                df['source_file'] = os.path.basename(path)
                return df
    return None
