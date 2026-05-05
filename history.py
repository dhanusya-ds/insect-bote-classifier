# history.py
import os
import pandas as pd
from datetime import datetime
from config import HISTORY_PATH, RESULTS_DIR

def save_to_history(filename, insect, confidence, severity):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_new = pd.DataFrame([{
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Filename": filename,
        "Insect": insect,
        "Confidence": f"{confidence:.1%}",
        "Severity": severity
    }])
    if os.path.exists(HISTORY_PATH):
        existing = pd.read_csv(HISTORY_PATH)
        df = pd.concat([existing, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(HISTORY_PATH, index=False)
    return df

def load_history():
    if os.path.exists(HISTORY_PATH):
        return pd.read_csv(HISTORY_PATH)
    return pd.DataFrame(columns=["Timestamp", "Filename", "Insect", "Confidence", "Severity"])

def clear_history():
    if os.path.exists(HISTORY_PATH):
        os.remove(HISTORY_PATH)