# 🛰️ Dream-in-Orbit: An AI That Simulates a Satellite

**Satellite Electrical Power Prediction and AI-Powered Simulation Using Machine Learning**

## 1. Project Overview

Dream-in-Orbit is an artificial intelligence and machine learning project designed to analyze, predict, and simulate satellite electrical power behavior using historical spacecraft telemetry data.

The project combines exploratory data analysis, data preprocessing, feature engineering, predictive modeling, and interactive simulation to demonstrate how data-driven techniques can support satellite power monitoring.

The primary objective is to develop an AI-based prototype capable of learning patterns from satellite telemetry and predicting future electrical power measurements.

The project also includes a separate hourly prediction deliverable covering multiple satellite power variables.

## 2. Problem Statement

Satellites rely on electrical power systems to support essential onboard operations, including communication, navigation, sensing, and onboard computing.

Changes in electrical power behavior can affect the reliability and efficiency of satellite operations.

Predicting future power behavior can help engineers understand operational trends, identify unusual patterns, and support informed power-management decisions.

Dream-in-Orbit investigates how machine learning can be used to model satellite power behavior from historical telemetry and demonstrate predictions through an interactive simulation.

## 3. Project Objectives

The main objectives of the project are to:

1. Explore and understand historical satellite telemetry datasets.
2. Identify patterns, trends, correlations, and anomalies in satellite electrical power measurements.
3. Clean and preprocess telemetry data for machine learning.
4. Develop relevant features from historical power measurements.
5. Train and compare machine learning models for satellite power prediction.
6. Evaluate predictive performance using appropriate regression metrics.
7. Develop an interactive AI-powered satellite power simulation.
8. Generate and validate a final hourly satellite power prediction file.
9. Document the methodology, results, and limitations in a reproducible GitHub project.

## 4. Dataset Description

The project uses historical satellite telemetry organized into two datasets:

- `annee1` — First observation period.
- `annee2` — Second observation period.

The telemetry includes electrical power measurements and additional operational information.

The original dataset directories contain files such as:

- `power.csv` — Satellite electrical power measurements.
- `dmop.csv`
- `evtf.csv`
- `ftl.csv`
- `ltdata.csv`
- `saaf.csv`

The analysis focuses on understanding and predicting satellite electrical power behavior.

### Data availability

The original telemetry datasets and some generated model artifacts are excluded from GitHub because of their size.

Users must obtain the required datasets separately and place them in the appropriate local directories before reproducing the complete workflow.

## 5. Technologies and Tools

| Category | Technologies |
|---|---|
| Programming language | Python |
| Development environment | Jupyter Notebook, Anaconda |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Visualization | Matplotlib |
| Interactive simulation | ipywidgets |
| Model serialization | Joblib |
| Version control | Git and GitHub |
| Archived model scripts | PyTorch, Scikit-learn |

## 6. Project Structure

```text
Dream-in-Orbit-an-AI-that-simulates-a-satellite/
│
├── archive/
│   ├── evaluate.py
│   ├── features.py
│   ├── predict.py
│   ├── train.py
│   ├── train_additional.py
│   ├── predict_additional.py
│   ├── make_validation.py
│   └── worldmodel.py
│
├── data/
│   ├── annee1/
│   ├── annee2/
│   └── processed/
│
├── notebooks/
│   ├── 01_Satellite_Data_Exploration.ipynb
│   ├── 02_Satellite_Data_Preprocessing.ipynb
│   ├── 03_Satellite_Power_Prediction.ipynb
│   └── 04_Satellite_Power_Simulation.ipynb
│
├── figures/
├── results/
├── scripts/
│
├── final_predictions.csv
├── README.md
└── .gitignore
```

**Note:** Some local directories and large data files are excluded from version control. The archived scripts are retained separately from the main notebook workflow.

## 7. Project Methodology

The project follows a structured machine learning workflow.

### Stage 1 — Exploratory Data Analysis

**Notebook:** `01_Satellite_Data_Exploration.ipynb`

This stage investigates the characteristics of the satellite telemetry datasets.

Activities include:

- Examining dataset structure and variables.
- Identifying missing values and data quality issues.
- Analyzing electrical power distributions.
- Investigating correlations between power measurements.
- Exploring temporal patterns and unusual observations.
- Visualizing satellite power behavior.

### Stage 2 — Data Preprocessing and Feature Engineering

**Notebook:** `02_Satellite_Data_Preprocessing.ipynb`

This stage prepares the telemetry data for machine learning.

The main forecasting target is `NPWD2721`, with `NPWD2802` used as an additional input variable.

Feature engineering includes:

- Current telemetry measurements.
- Historical lagged measurements.
- Rolling means.
- Rolling standard deviations.

The forecasting horizon is **288 seconds (approximately 4.8 minutes)**.

The data is divided chronologically into:

| Dataset split | Percentage |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

Chronological splitting helps preserve temporal order and reduce data leakage.

### Stage 3 — Machine Learning Model Development

**Notebook:** `03_Satellite_Power_Prediction.ipynb`

Three forecasting approaches were evaluated:

1. Persistence baseline.
2. Ridge Regression.
3. Histogram Gradient Boosting Regressor.

The models were evaluated using:

- Mean Absolute Error (MAE).
- Root Mean Squared Error (RMSE).
- Coefficient of Determination (R²).

Histogram Gradient Boosting was selected based on validation performance for both datasets.

### Stage 4 — AI-Powered Satellite Power Simulation

**Notebook:** `04_Satellite_Power_Simulation.ipynb`

The simulation demonstrates how a trained machine learning model can predict future satellite power behavior using historical telemetry observations.

The simulation includes:

- Actual versus predicted satellite power visualization.
- An interactive slider for selecting simulation steps.
- Prediction error monitoring.
- Power-state classification into LOW, MEDIUM, and HIGH categories.

The simulation uses held-out historical observations and is intended as a data-driven demonstration rather than a complete orbital physics simulator.

## 8. Machine Learning Results

### Histogram Gradient Boosting — Test Performance

| Dataset | MAE | RMSE | R² |
|---|---:|---:|---:|
| `annee1` | 0.0282 | 0.0547 | 0.7634 |
| `annee2` | 0.0373 | 0.0701 | 0.6841 |

### Persistence Baseline — Test Performance

| Dataset | MAE | RMSE | R² |
|---|---:|---:|---:|
| `annee1` | 0.0207 | 0.0573 | 0.7404 |
| `annee2` | 0.0287 | 0.0720 | 0.6670 |

The selected Histogram Gradient Boosting models achieved lower RMSE and higher R² than the persistence baseline on both test datasets.

However, the persistence baseline achieved lower MAE. This highlights the importance of evaluating models using multiple performance metrics.

## 9. Interactive Simulation Results

The interactive simulation was evaluated using 1,000 held-out observations from the first dataset.

| Metric | Result |
|---|---:|
| Simulation steps | 1,000 |
| Prediction horizon | 288 seconds |
| MAE | 0.0269 |
| RMSE | 0.0555 |
| R² | 0.7557 |

The simulation allows users to examine individual prediction steps, compare actual and predicted power measurements, and monitor changes in the predicted power state.

## 10. Final Prediction Deliverable

The project includes a validated output file:

**`final_predictions.csv`**

This file contains hourly satellite electrical power predictions across 33 power variables.

### File validation

| Property | Result |
|---|---|
| Filename | `final_predictions.csv` |
| Number of rows | 16,319 |
| Number of columns | 34 |
| Power variables | 33 |
| Timestamp column | `ut_ms` |
| Missing values | 0 |
| Duplicate timestamps | 0 |
| Timestamp ordering | Chronological |
| Time interval | 1 hour |

The prediction timestamps cover the period from July 17, 2010, to May 26, 2012, in UTC.

### Evaluation

The final prediction file was evaluated using the project's archived evaluation script.

**Reported RMSE: 0.111621**

This evaluation concerns the separate hourly multi-variable prediction deliverable and should not be directly compared with the notebook forecasting metrics because the prediction tasks and evaluation setups differ.

## 11. Installation and Setup

### Prerequisites

- Python
- Anaconda or another compatible Python environment
- Jupyter Notebook
- Required Python libraries
- Original satellite telemetry datasets for full reproduction

### Clone the repository

```bash
git clone https://github.com/damisar/Dream-in-Orbit-an-AI-that-simulates-a-satellite.git
```

Navigate into the project directory:

```bash
cd Dream-in-Orbit-an-AI-that-simulates-a-satellite
```

### Create a Python environment

```bash
conda create -n dream-orbit python=3.11 -y
conda activate dream-orbit
```

### Install the main notebook dependencies

```bash
python -m pip install numpy pandas scikit-learn matplotlib seaborn scipy joblib ipywidgets jupyter ipykernel
```

The archived training and prediction scripts also require PyTorch.

### Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the notebooks in the following order:

1. `01_Satellite_Data_Exploration.ipynb`
2. `02_Satellite_Data_Preprocessing.ipynb`
3. `03_Satellite_Power_Prediction.ipynb`
4. `04_Satellite_Power_Simulation.ipynb`

The complete notebook workflow requires the original telemetry data and locally generated preprocessing/model artifacts.

## 12. Project Limitations

The current project has several limitations:

- The interactive simulation is based on historical telemetry rather than real-time satellite data.
- The simulation does not reproduce complete satellite orbital dynamics or physical power-system behavior.
- Forecasting performance may vary across datasets and operating conditions.
- Large datasets and some trained models are not included in the public repository.
- The archived model-training pipeline remains a separate workflow and has not been fully reproduced as part of the final notebook implementation.

These limitations provide opportunities for future development.

## 13. Future Improvements

Potential improvements include:

1. Exploring advanced deep learning architectures for satellite power forecasting.
2. Implementing automated anomaly detection.
3. Incorporating additional telemetry variables and operational conditions.
4. Improving model robustness across different observation periods.
5. Developing more realistic satellite power-system simulations.
6. Extending the interactive monitoring interface.
7. Improving reproducibility through automated training and evaluation scripts.

## 14. Conclusion

Dream-in-Orbit demonstrates how artificial intelligence and machine learning can be applied to historical satellite telemetry for electrical power analysis, forecasting, and interactive monitoring.

The project integrates data exploration, preprocessing, predictive modeling, performance evaluation, and simulation into a structured workflow.

The resulting prototype provides an educational and practical demonstration of data-driven satellite power prediction. The validated hourly prediction deliverable further illustrates the application of machine learning techniques to multiple satellite electrical power variables.

Although the current implementation is not a complete physical satellite simulator, it establishes a foundation for future research into intelligent satellite monitoring, forecasting, and operational decision support.

---

**Project:** Dream-in-Orbit: An AI That Simulates a Satellite

**Repository:** https://github.com/damisar/Dream-in-Orbit-an-AI-that-simulates-a-satellite

**Development environment:** Python, Anaconda, Jupyter Notebook

**Version control:** Git and GitHub