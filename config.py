# config.py
import os

# Set target paths
DATA_DIR = "flat_training"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sepsis task requirements
PREDICTION_WINDOW = 6  # Focus window: 6 hours prior to physician clinical onset
RANDOM_SEED = 42

# Relevant core features tracking vital signs and laboratory results
FEATURES = [
    'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
    'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
    'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Glucose'
]