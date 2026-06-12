# Clinical Early-Warning Pipeline for Sepsis Onset Prediction

A modular, production-grade machine learning pipeline engineered to predict sepsis onset **6 hours prior to clinical diagnosis** using highly irregular, high-missingness ICU time-series data. The architecture incorporates robust strategies for handling informatively missing clinical features, mitigating documentational label noise, preventing temporal data leakage, and calibrating outputs for reliable clinical risk assessment.

---

## 📂 Repository Architecture

🛠️ Key Methodological Solutions
1. Handling Irregular & Missing Time-Series

ICU missing data density can exceed 60% per patient stay, where missingness itself is often highly informative. Rather than using biasing global statistical means, the pipeline enforces a Causal Last Observation Carried Forward (LOCF) approach strictly within individual patient boundaries. Measurements flow exclusively past-to-future, preventing backwards temporal data corruption.
2. Global Label Smoothing for Noise Mitigation

Physician diagnostic charting in electronic health records routinely suffers from documentational entry lag. Training models on hard binary boundaries introduces massive gradient penalty overhead for minor timing discrepancies. We resolve this by mapping targets to soft margins (α=0.1), smoothing vectors to [0.05,0.95] to force the model to prioritize underlying physiology over inaccurate human time-logging constraints.
3. Strict Prevention of Data Leakage

To eliminate over-optimistic performance scaling, target windows undergo an early horizon transformation. Patient vectors are strictly isolated at split-time, ensuring rolling properties never blend look-ahead information into historical prediction sequences.
4. Platt Scaling Probability Calibration

Gradient-boosted ensembles inherently produce uncalibrated continuous scores rather than true clinical probabilities. To allow reliable decision mapping, the engine feeds raw outputs into a post-hoc Platt Scaling calibrator (via Logistic Regression), guaranteeing that an output score maps precisely to true historical event frequencies.
🚀 Setup & Execution Guide
1. Environment Initialization

Isolate your development runtime using a local virtual environment to prevent package version conflicts:
Bash

# Create and activate environment
python3 -m venv venv
source venv/bin/activate

# Install core performance dependencies
pip install pandas numpy scikit-learn lightgbm matplotlib

2. Dataset Alignment

Ensure your unzipped PhysioNet Sepsis Challenge 2019 patient files are stored flatly in the root directory structure:
Bash

Sepsis_Prediction/flat_training/patient_000001.psv

The project is strictly modularized into distinct operational layers to maintain clean separation of concerns and maximize reproducibility:

```text
Sepsis_Prediction/
├── flat_training/          # Local directory for PhysioNet .psv source files (Git ignored)
├── outputs/                # Automated evaluation graph exports
│   └── screening_metrics_report.png
├── config.py               # Centralized hyperparameters and feature boundaries
├── data_cleaner.py         # Causal imputation and 6-hour horizon target engineering
├── noise_smoothing.py      # Regularization layer for delayed physician logging
└── train_eval.py           # Core controller for execution, calibration, and evaluation
