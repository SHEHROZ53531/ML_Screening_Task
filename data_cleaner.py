# data_cleaner.py
import os
import pandas as pd
import numpy as np
import config

def process_patient_file(file_path):
    """Parses a patient matrix, fills irregular missing signals,

    and flags the early prediction warning target.
    """
    df = pd.read_csv(file_path, sep='|')
    if df.empty:
        return None
    
    # 1. Missingness handling: Forward-fill values to represent last known clinical state
    df_clean = df[config.FEATURES].ffill().bfill().fillna(0)
    
    # 2. Strict Target Horizon Transformation (6 hours pre-onset window)
    clinical_labels = df['SepsisLabel'].values
    early_target = np.zeros(len(clinical_labels))
    
    # Locate actual documented sepsis presentation hour
    sepsis_indices = np.where(clinical_labels == 1)[0]
    
    if len(sepsis_indices) > 0:
        actual_onset_hour = sepsis_indices[0]
        # Map out the exact prediction target zone 6 hours ahead
        warning_start = max(0, actual_onset_hour - config.PREDICTION_WINDOW)
        early_target[warning_start:actual_onset_hour] = 1
        
    df_clean['Target'] = early_target
    return df_clean

def load_and_prepare_dataset(max_files=4000):
    """Scans the flat database directory and compiles chunks of arrays."""
    if not os.path.exists(config.DATA_DIR):
        raise FileNotFoundError(f"Configured directory '{config.DATA_DIR}' cannot be verified.")
        
    files = [os.path.join(config.DATA_DIR, f) for f in os.listdir(config.DATA_DIR) if f.endswith('.psv')]
    print(f"Directory Validation: Located {len(files)} records in target path.")
    
    # Bound iteration limits to maintain efficient local compilation execution
    run_limit = min(len(files), max_files)
    print(f"Assembling structural matrix using a subset of {run_limit} patients...")
    
    patient_matrices = []
    for f in files[:run_limit]:
        cleaned_df = process_patient_file(f)
        if cleaned_df is not None:
            patient_matrices.append(cleaned_df)
            
    if not patient_matrices:
        raise ValueError("Pipeline structure execution interrupted: Zero valid files processed.")
        
    return pd.concat(patient_matrices, ignore_index=True)