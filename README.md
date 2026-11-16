# 🛰️ Dream-in-Orbit: An AI That Simulates a Satellite

## 1. Project Overview

**Dream-in-Orbit** is an AI-powered satellite power forecasting and simulation project that uses historical satellite telemetry to predict future electrical power measurements and demonstrate intelligent monitoring capabilities.

The project combines data exploration, time-series preprocessing, machine learning, and interactive simulation to investigate how artificial intelligence can support satellite power monitoring and operational analysis.

Using two years of satellite telemetry data, the system learns relationships between electrical power measurements and predicts the future value of the `NPWD2721` variable approximately **288 seconds (4.8 minutes) ahead**.

The developed prototype integrates a trained machine-learning model into an interactive simulation environment, allowing users to examine predicted power behavior, compare forecasts with actual observations, and monitor predicted power conditions.

### Project Scope

The current implementation focuses on **satellite electrical power prediction and monitoring** using historical telemetry.

It is an offline, data-driven simulation rather than a complete orbital physics simulator or a live satellite monitoring system.

## 2. Project Objectives

The primary objective of Dream-in-Orbit is to develop an AI-powered prototype capable of predicting satellite electrical power behavior and demonstrating its application in an interactive monitoring simulation.

### Specific Objectives

1. **Explore satellite telemetry data:** Analyze historical electrical power measurements to identify distributions, correlations, temporal patterns, and data-quality issues.

2. **Prepare data for machine learning:** Clean and transform satellite telemetry, engineer time-dependent features, and establish chronological training, validation, and testing datasets.

3. **Develop predictive AI models:** Implement and evaluate persistence, Ridge Regression, and Histogram Gradient Boosting approaches for short-term satellite power forecasting.

4. **Identify a suitable forecasting model:** Compare model performance using MAE, RMSE, and R², selecting a model based on validation performance.

5. **Develop an interactive satellite power simulation:** Integrate the selected forecasting model into a simulation that reproduces historical telemetry observations and predicts future power measurements.

6. **Monitor predicted power conditions:** Classify forecast power levels into LOW, MEDIUM, and HIGH categories and visualize prediction accuracy.

7. **Evaluate system performance:** Assess forecasting reliability, identify limitations, and establish a foundation for future improvements.

## 3. Project Architecture and Workflow

Dream-in-Orbit follows a modular, four-stage workflow that transforms historical satellite telemetry into an AI-powered satellite power prediction and monitoring simulation.

### 3.1 System Architecture

```text
              HISTORICAL SATELLITE TELEMETRY
                     (annee1, annee2)
                             |
                             v
                01 — DATA EXPLORATION
                 Data quality assessment
                 Statistical analysis
                 Temporal pattern analysis
                             |
                             v
                02 — DATA PREPROCESSING
                 Data cleaning
                 Feature engineering
                 Chronological data splitting
                 Feature scaling
                             |
                             v
                03 — AI POWER PREDICTION
                 Persistence baseline
                 Ridge Regression
                 Histogram Gradient Boosting
                 Model evaluation and selection
                             |
                             v
                TRAINED FORECASTING MODEL
                             |
                             v
                04 — POWER SIMULATION
                 Historical telemetry replay
                 Future power prediction
                 Actual vs. predicted comparison
                 Interactive monitoring
                 LOW / MEDIUM / HIGH classification
                             |
                             v
                PERFORMANCE EVALUATION
                 MAE | RMSE | R²
```

### 3.2 Notebook Organization

| Notebook | Description |
|---|---|
| `01_Satellite_Data_Exploration.ipynb` | Explores satellite telemetry, evaluates data quality, and investigates statistical and temporal patterns. |
| `02_Satellite_Data_Preprocessing.ipynb` | Cleans the telemetry, creates forecasting features, and prepares chronological training, validation, and test datasets. |
| `03_Satellite_Power_Prediction.ipynb` | Develops, compares, evaluates, and exports machine-learning forecasting models. |
| `04_Satellite_Power_Simulation.ipynb` | Integrates the selected model into an interactive satellite power prediction and monitoring simulation. |

### 3.3 Workflow Description

The project begins with two years of historical satellite telemetry. Exploratory analysis identifies relevant power measurements and supports the selection of variables for forecasting.

The preprocessing stage transforms these measurements into model-ready datasets, including lagged observations and rolling statistical features. Chronological data partitions are used to evaluate forecasting performance while reducing the risk of temporal data leakage.

The prediction stage evaluates multiple forecasting approaches and selects Histogram Gradient Boosting based on validation performance.

Finally, the simulation stage applies the trained model to held-out historical observations, compares predicted and actual future power measurements, and provides interactive monitoring of forecast power conditions.

The simulation is data-driven and operates offline; it does not model satellite orbital mechanics or communicate with an operational spacecraft.

## 4. Dataset Description and Feature Engineering

### 4.1 Dataset Overview

Dream-in-Orbit uses two historical satellite telemetry datasets, identified as `annee1` and `annee2`.

The datasets contain time-series measurements describing satellite electrical power behavior. They are analyzed and processed separately to investigate forecasting performance across different observation periods.

The original telemetry contains 33 power-related variables. Following exploratory analysis, two measurements were selected as the primary inputs for the forecasting task:

- **`NPWD2721`:** Primary electrical power variable and the basis of the prediction target.
- **`NPWD2802`:** Additional electrical power variable used to provide supporting information.

The forecasting objective is to predict the future value of `NPWD2721` approximately **288 seconds (4.8 minutes) ahead**.

### 4.2 Feature Engineering

To capture temporal relationships, the preprocessing pipeline generates 16 input features from the two selected power variables.

| Feature category | Description | Number of features |
|---|---|---:|
| Current measurements | Current values of `NPWD2721` and `NPWD2802` | 2 |
| Lagged measurements | Previous values at 60, 120, and 300 seconds | 6 |
| Rolling averages | Rolling means over 120- and 300-second windows | 4 |
| Rolling standard deviations | Rolling variability over 120- and 300-second windows | 4 |
| **Total** | **Model input features** | **16** |

These features provide information about recent power levels, historical behavior, and short-term variability.

### 4.3 Data Preprocessing

The preprocessing workflow includes:

1. Data-quality assessment and cleaning.
2. Chronological ordering and timestamp handling.
3. Generation of lagged and rolling statistical features.
4. Construction of future prediction targets.
5. Filtering of invalid observations and unsuitable temporal gaps.
6. Chronological separation into training, validation, and testing datasets.
7. Feature scaling using parameters fitted on training data only.

The chronological partitioning strategy allocates approximately:

- **70%** of observations to training.
- **15%** to validation.
- **15%** to testing.

This approach supports evaluation on later observations while reducing the risk of temporal data leakage.

### 4.4 Data Availability

The original telemetry datasets, processed NumPy arrays, and serialized trained models are excluded from the GitHub repository because of their storage requirements.

The repository contains the Jupyter notebooks and project code. Reproducing the full workflow requires access to the original satellite telemetry datasets.

Users with access to the required datasets can execute the notebooks in their numerical order to regenerate the preprocessing outputs, train the forecasting models, and run the simulation.

## 5. Machine Learning Models and Forecasting Results

### 5.1 Forecasting Approaches

Dream-in-Orbit evaluates three approaches for predicting satellite electrical power approximately 288 seconds ahead.

**1. Persistence Baseline**

The persistence baseline assumes that the future power measurement will be equal to the current measurement. It provides a simple reference against which machine-learning models can be evaluated.

**2. Ridge Regression**

Ridge Regression is a regularized linear regression algorithm that models relationships between the engineered input features and the future power target.

**3. Histogram Gradient Boosting**

Histogram Gradient Boosting is a tree-based machine-learning algorithm capable of learning nonlinear relationships between satellite power measurements and engineered temporal features.

### 5.2 Model Evaluation Metrics

The forecasting models are evaluated using three performance metrics:

- **Mean Absolute Error (MAE):** Measures the average absolute difference between predicted and actual values.
- **Root Mean Squared Error (RMSE):** Measures prediction error while placing greater emphasis on larger errors.
- **Coefficient of Determination (R²):** Measures the proportion of variation in the target explained by the model.

Lower MAE and RMSE values indicate better predictive accuracy, while higher R² values generally indicate better explanatory performance.

### 5.3 Final Test Results

The following table summarizes the selected Histogram Gradient Boosting model's performance on the independent test datasets.

| Evaluation metric | `annee1` | `annee2` |
|---|---:|---:|
| MAE | 0.0282 | 0.0373 |
| RMSE | 0.0547 | 0.0701 |
| R² | 0.7634 | 0.6841 |

For comparison, the persistence baseline achieved the following test results:

| Evaluation metric | `annee1` | `annee2` |
|---|---:|---:|
| MAE | 0.0207 | 0.0287 |
| RMSE | 0.0573 | 0.0720 |
| R² | 0.7404 | 0.6670 |

### 5.4 Model Selection

Histogram Gradient Boosting achieved the lowest validation RMSE among the evaluated forecasting approaches and was selected for the satellite power simulation.

On the test datasets, it reduced RMSE relative to the persistence baseline by approximately:

- **4.5% for `annee1`**
- **2.6% for `annee2`**

However, the persistence baseline achieved lower MAE on both test datasets. Therefore, the selected model should not be interpreted as outperforming the baseline across every evaluation metric.

The results demonstrate that Histogram Gradient Boosting provides modest improvements in RMSE and R², while also revealing opportunities for further improvement.

### 5.5 Forecasting Limitations

The current forecasting approach has several limitations:

- Forecasting is restricted to a single prediction horizon of approximately 288 seconds.
- Sudden changes in satellite power behavior may produce larger prediction errors.
- Improvements over the persistence baseline are relatively modest.
- The model is trained using selected electrical power variables rather than a complete representation of satellite operating conditions.

Future work may investigate additional telemetry variables, alternative prediction horizons, and sequence-based forecasting architectures.

## 6. Interactive Satellite Power Simulation

### 6.1 Simulation Overview

Dream-in-Orbit integrates the selected Histogram Gradient Boosting forecasting model into an interactive satellite power monitoring simulation.

The simulation uses previously unseen observations from the `annee1` test dataset to demonstrate how machine learning can predict future satellite electrical power measurements.

The simulation evaluates **1,000 observations**, maintaining the original prediction horizon of approximately **288 seconds (4.8 minutes)**.

### 6.2 Simulation Features

The interactive simulation provides the following capabilities:

1. **Satellite Power Forecasting:** Predicts future values of the `NPWD2721` electrical power variable using the trained forecasting model.

2. **Actual vs. Predicted Visualization:** Displays observed and predicted future power measurements to illustrate forecasting behavior.

3. **Interactive Simulation Control:** Uses an `ipywidgets` slider to allow users to explore individual simulation steps.

4. **Power Monitoring:** Displays the actual future power value, predicted value, and absolute prediction error for a selected observation.

5. **Power Level Classification:** Categorizes predicted power conditions into LOW, MEDIUM, and HIGH levels using data-derived thresholds.

6. **Performance Evaluation:** Calculates forecasting accuracy across the simulation observations.

### 6.3 Simulation Performance

The selected Histogram Gradient Boosting model achieved the following results across the 1,000 simulated observations:

| Metric | Simulation Result |
|---|---:|
| Mean Absolute Error (MAE) | 0.0269 |
| Root Mean Squared Error (RMSE) | 0.0555 |
| Coefficient of Determination (R²) | 0.7557 |
| Simulation observations | 1,000 |
| Forecast horizon | 288 seconds |

These results indicate that the forecasting model captures a substantial proportion of the observed variation in future satellite power measurements.

However, prediction errors remain, particularly during abrupt changes in power behavior.

### 6.4 Simulation Scope and Limitations

The current implementation is an **offline, data-driven satellite power simulation** based on historical telemetry.

It does not currently simulate orbital trajectories, spacecraft attitude, solar panel physics, battery charging dynamics, or real-time satellite communication.

The LOW, MEDIUM, and HIGH categories are relative classifications derived from the available data. They should not be interpreted as verified spacecraft safety thresholds.

The simulation demonstrates the potential application of machine learning in satellite power monitoring and provides a foundation for future system development.

## 7. Installation and Usage

### 7.1 System Requirements

The project was developed using:

- Python 3.11.18
- Anaconda / Conda
- Jupyter Notebook
- NumPy, Pandas, Matplotlib, and Seaborn
- Scikit-learn and SciPy
- Joblib
- ipywidgets

The required Python package versions are listed in `requirements.txt`.

### 7.2 Clone the Repository

Open Anaconda Prompt or a terminal and clone the GitHub repository:

```bash
git clone https://github.com/damisar/Dream-in-Orbit-an-AI-that-simulates-a-satellite.git
```

Navigate to the project directory:

```bash
cd Dream-in-Orbit-an-AI-that-simulates-a-satellite
```

### 7.3 Create the Python Environment

Create a dedicated Conda environment using Python 3.11:

```bash
conda create -n dream-orbit python=3.11 -y
```

Activate the environment:

```bash
conda activate dream-orbit
```

Install the required Python libraries:

```bash
python -m pip install -r requirements.txt
```

Install Jupyter Notebook if it is not already available:

```bash
python -m pip install notebook
```

Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name dream-orbit --display-name "Python (Dream Orbit)"
```

### 7.4 Prepare the Satellite Telemetry Data

The original satellite telemetry datasets and generated model artifacts are not included in this repository.

To reproduce the complete workflow, obtain the original `annee1` and `annee2` datasets and place them in the data directories expected by the notebooks.

The preprocessing notebook generates the processed datasets, while the prediction notebook trains and exports the machine-learning models required by the simulation.

### 7.5 Launch Jupyter Notebook

From the project directory, run:

```bash
jupyter notebook
```

Select the **Python (Dream Orbit)** kernel when opening the notebooks.

### 7.6 Execute the Project

Run the notebooks in the following order:

1. `01_Satellite_Data_Exploration.ipynb` — Explore the satellite telemetry datasets.
2. `02_Satellite_Data_Preprocessing.ipynb` — Prepare the forecasting features and datasets.
3. `03_Satellite_Power_Prediction.ipynb` — Train and evaluate the forecasting models.
4. `04_Satellite_Power_Simulation.ipynb` — Run the interactive satellite power simulation.

The notebooks should be executed sequentially because later stages depend on datasets and model artifacts generated by earlier stages.

### 7.7 Interactive Simulation

In Notebook 04, execute the simulation cells and use the interactive slider to explore the forecast results.

The interface displays predicted future power measurements, actual observations, prediction errors, and relative power-level classifications.

**Note:** A successful `pip check` in the original development environment does not guarantee that every package version will install successfully on another computer. Installation compatibility should be verified separately.

## 8. Project Limitations and Future Improvements

### 8.1 Current Limitations

Although Dream-in-Orbit demonstrates the application of artificial intelligence to satellite power forecasting and monitoring, several limitations remain.

1. **Historical Data Dependence:** The forecasting models are trained and evaluated using historical satellite telemetry rather than live satellite measurements.

2. **Limited Input Variables:** The current implementation uses two electrical power variables and their engineered temporal features. Other operating conditions that may influence satellite power behavior are not explicitly represented.

3. **Single Forecast Horizon:** The models predict power approximately 288 seconds ahead. Their performance at shorter or longer forecasting horizons has not been evaluated.

4. **Forecasting Accuracy:** Histogram Gradient Boosting improves RMSE and R² relative to the persistence baseline, but the improvements are modest. The persistence baseline also achieves lower MAE.

5. **Simplified Power Monitoring:** LOW, MEDIUM, and HIGH classifications are based on relative thresholds derived from the available data, not validated spacecraft operational limits.

6. **Offline Simulation:** The simulation replays historical telemetry and generates model predictions. It does not simulate satellite orbital dynamics, spacecraft attitude, or physical power-system behavior.

7. **Data Availability:** The original telemetry datasets and trained model files are not included in the public repository, limiting immediate reproducibility without access to the source data.

### 8.2 Future Improvements

Potential extensions of Dream-in-Orbit include:

1. **Advanced Time-Series Forecasting:** Investigate sequence-based models such as Long Short-Term Memory (LSTM) networks, Temporal Convolutional Networks (TCNs), and other forecasting architectures.

2. **Multi-Horizon Prediction:** Extend the forecasting framework to predict satellite power at multiple future time intervals.

3. **Additional Telemetry Features:** Incorporate more electrical, thermal, and operational measurements to improve the representation of satellite conditions.

4. **Anomaly Detection:** Develop AI-based methods to identify unusual power behavior and potential operational anomalies.

5. **Physics-Based Satellite Simulation:** Integrate simplified models of orbital illumination, solar energy generation, battery behavior, and spacecraft electrical loads.

6. **Real-Time Monitoring:** Explore integration with live telemetry streams or a simulated real-time data source.

7. **Improved Visualization:** Develop a dedicated dashboard for monitoring satellite power forecasts, system status, and prediction uncertainty.

8. **Model Robustness Evaluation:** Assess forecasting performance across additional datasets, operating conditions, and time periods.

### 8.3 Long-Term Vision

The long-term vision of Dream-in-Orbit is to evolve from an offline satellite power forecasting prototype into a more comprehensive AI-assisted satellite monitoring and simulation platform.

Such a platform could combine data-driven forecasting, physical satellite models, anomaly detection, and interactive visualization to support research into intelligent spacecraft operations.

These capabilities represent future development opportunities and are not part of the current implementation.

## 9. Project Conclusion

Dream-in-Orbit demonstrates how artificial intelligence and historical satellite telemetry can be combined to develop a data-driven satellite electrical power forecasting and monitoring prototype.

The project successfully integrates four major components: exploratory data analysis, time-series preprocessing, machine-learning forecasting, and interactive simulation.

Three forecasting approaches were evaluated: Persistence, Ridge Regression, and Histogram Gradient Boosting. Histogram Gradient Boosting was selected based on validation RMSE and demonstrated modest improvements in test RMSE and R² compared with the persistence baseline.

The selected forecasting model was subsequently integrated into an interactive simulation that predicts satellite electrical power approximately 288 seconds ahead, visualizes actual and predicted measurements, and classifies forecast power conditions.

Although the current system is limited to offline telemetry-based simulation, it establishes a practical foundation for investigating more advanced AI-assisted satellite monitoring techniques.

Overall, Dream-in-Orbit illustrates both the potential and the limitations of machine learning for short-term satellite power forecasting, while providing a modular framework for future research and development.