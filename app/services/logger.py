# === services/logger.py ===
import pandas as pd
from datetime import datetime
import os

LOG_PATH = "../resources/logs/chat_logs.csv"  # Update path as needed

def log_interaction(username, department, query, response, confidence):
    log_data = {
        "timestamp": datetime.now(),
        "username": username,
        "department": department,
        "query": query,
        "response": response,
        "confidence": confidence
    }

    df = pd.DataFrame([log_data])
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    if os.path.exists(LOG_PATH):
        df.to_csv(LOG_PATH, mode='a', index=False, header=False)
    else:
        df.to_csv(LOG_PATH, mode='w', index=False, header=True)
