# train_eval.py
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve
import lightgbm as lgb

import config
import data_cleaner
import noise_smoothing

def main():
    # 1. Pipeline ingestion
    print("Step 1: Running chronological tabular clean up loops...")
    dataset = data_cleaner.load_and_prepare_dataset(max_files=4000)
    
    X = dataset[config.FEATURES]
    y = dataset['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_SEED
    )
    
    # 2. Label Noise Mitigation
    print("Step 2: Regularizing diagnostic vectors using soft smoothing...")
    y_train_smoothed = noise_smoothing.apply_label_smoothing(y_train, factor=0.1)
    
    # 3. Model Engine Optimization
    print("Step 3: Engineering gradient boosted tree ensembles...")
    train_payload = lgb.Dataset(X_train, label=y_train_smoothed)
    model_settings = {
        'objective': 'regression',
        'metric': 'rmse',
        'seed': config.RANDOM_SEED,
        'learning_rate': 0.05,
        'verbose': -1
    }
    model = lgb.train(model_settings, train_payload, num_boost_round=100)
    
    # 4. Calibration Engine Mapping (Platt Scaling)
    print("Step 4: Mapping raw score outputs to calibrated risk scales...")
    raw_train_preds = model.predict(X_train)
    raw_test_preds = model.predict(X_test)
    
    calibrator = LogisticRegression()
    calibrator.fit(raw_train_preds.reshape(-1, 1), y_train)
    calibrated_risk = calibrator.predict_proba(raw_test_preds.reshape(-1, 1))[:, 1]
    
    # 5. Threshold Tuning & Assessment Metrics
    clinical_safety_threshold = 0.35
    binary_predictions = (calibrated_risk >= clinical_safety_threshold).astype(int)
    
    print("\n=== COMPLETE PIPELINE PERFORMANCE METRICS ===")
    print(classification_report(y_test, binary_predictions))
    
    # 6. Evaluation Deliverables Generation
    print("Step 5: Visualizing evaluation metrics...")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left subplot: Confusion Matrix
    ConfusionMatrixDisplay(confusion_matrix(y_test, binary_predictions)).plot(ax=ax[0], cmap='Blues')
    ax[0].set_title(f'Confusion Matrix (Decision Limit={clinical_safety_threshold})')
    
    # Right subplot: Reliability Diagram
    observed_rate, predicted_rate = calibration_curve(y_test, calibrated_risk, n_bins=10)
    ax[1].plot(predicted_rate, observed_rate, marker='o', label='Calibrated Prediction Engine')
    ax[1].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Ideal Clinical Reality')
    ax[1].set_title('Probability Calibration Reliability Diagram')
    ax[1].set_xlabel('Predicted Sepsis Risk Level')
    ax[1].set_ylabel('True Observed Sepsis Rate')
    ax[1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_DIR}/screening_metrics_report.png")
    print(f"\nAll verification files saved successfully inside: {config.OUTPUT_DIR}/screening_metrics_report.png")
    plt.show()

if __name__ == "__main__":
    main()