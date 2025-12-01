# Baseline_JSP_Portafolio

Machine learning baselines for automatic solver selection in Job Shop Scheduling instances.
The core logic lives in the Jupyter notebook [`Baseline_JSP_Portafolio.ipynb`](Baseline_JSP_Portafolio.ipynb).

---

## 1. Project overview

This repository contains a reproducible baseline for **algorithm selection** in a **Job Shop Scheduling Problem (JSP)** portfolio.

Given a set of JSP instances described by meta‑features (number of jobs, number of machines, time‑window characteristics, processing and energy statistics, etc.) and performance results of several solvers, the notebook trains models that:

- Learn which solver is the **best** for each instance.
- Predict, for a new instance, which solver is likely to perform best.

Currently, the portfolio includes (depending on the dataset):

- Exact solvers: CPLEX, Gurobi, Gecode
- Metaheuristics: Genetic Algorithm (GA), Particle Swarm Optimization (PSO)

The notebook supports two scenarios:

1. **Full portfolio** including PSO.
2. **Reduced portfolio** without PSO (e.g., when PSO results are missing).

---

## 2. Repository contents

- Main notebook: [`Baseline_JSP_Portafolio.ipynb`](Baseline_JSP_Portafolio.ipynb)
  End‑to‑end pipeline:
  - Data loading from an uploaded Excel file.
  - Cleaning and standardization of column names.
  - Type conversion and basic quality checks.
  - Construction of labels (`best_solver`) from solver performance columns.
  - Train/validation/test splits and preprocessing.
  - Baseline models:
    - Multilayer Perceptron (MLP) classifier using TensorFlow / Keras.
    - XGBoost multi‑class classifier.
  - Evaluation (accuracy, top‑k accuracy, classification reports, confusion matrices).
  - Basic feature importance analysis for XGBoost.
  - Utility code for per‑instance explanations and predictions.

- README: [`README.md`](README.md) (this file).

The notebook is designed to run **directly in Google Colab**.

---

## 3. Input data

The notebook expects an **Excel (.xlsx)** file with at least the following types of columns:

### 3.1. Instance meta‑features (examples)

The exact set of columns is driven by the uploaded file, but the notebook is prepared to work with features such as:

- `cxd`: Seed or configuration identifier used to generate the instance.
- `jobs`: Number of jobs (orders).
- `machines`: Number of machines (resources).
- `rddd`: Type of time windows:
  - `0`: no windows
  - `1`: per job
  - `2`: per operation
- `speed`: Number of available speed levels.
- `max_makespan`, `min_makespan`: Simple upper and lower bounds on total completion time.
- `max_sum_energy`, `min_sum_energy`: Upper and lower bounds on total energy consumption.
- `max_tardiness`: Maximum theoretical tardiness; `-1` if there are no due dates.
- `min_window`, `max_window`, `mean_window`: Tightest, widest, and average time‑window size; `-1` if no windows exist.
- `overlap`: Degree of overlap between time windows (0 = low, 1 = high).
- `max_processing_time_value`, `min_processing_time_value`, `mean_processing_time_value`: Statistics of operation durations.
- `max_energy_value`, `min_energy_value`, `mean_energy_value`: Statistics of per‑operation energy consumption.
- `problema`: Textual identifier of the instance (metadata).

All of these are treated as numeric features where possible; the notebook includes robust conversion from strings with commas, etc.

### 3.2. Solver performance columns

The dataset must contain **at least two** of the following solver columns:

- `cplex`
- `gurobi`
- `gecode`
- `ga`
- `pso` (optional in the reduced‑portfolio setting)

Each column stores a performance **score or objective value** for that solver.
Depending on the metric, **lower or higher values may be better**:

- There is code to define the best solver assuming *lower is better*.
- There is an alternate cell (commented in the notebook) for the case where *higher is better*.

The notebook then creates:

- `best_solver_idx`: integer label of the best solver.
- `best_solver_name`: textual name of the best solver.

---

## 4. Preprocessing pipeline

The preprocessing steps implemented in [`Baseline_JSP_Portafolio.ipynb`](Baseline_JSP_Portafolio.ipynb) are:

1. **Upload & sheet selection** (Colab):
   - The user uploads an `.xlsx` file.
   - The notebook shows available sheets and defaults to the first one.

2. **Column cleaning**:
   - Strips extra spaces from names.
   - Drops automatically generated `Unnamed: *` index columns from Excel.
   - Replaces spaces with underscores for easier programmatic access.

3. **Numeric coercion**:
   - For a predefined list of numeric columns, attempts to convert values to floats.
   - Handles string numbers with commas by replacing `,` with `.`.
   - Reports the number of `NaN` values per column after conversion.

4. **Label creation**:
   - For the selected set of solver columns, replaces missing performance values with:
     - `+∞` when **smaller is better** (so missing values are always worst).
     - or `−∞` when **larger is better** (alternative cell).
   - Finds the index of the best solver for each instance and stores it as the target label.

5. **Feature / target split**:
   - Drops identifier and solver columns.
   - Keeps only numeric meta‑features as `X`.
   - Uses `best_solver_idx` as `y`.

6. **Row filtering**:
   - Removes rows where **all** feature values are `NaN`.

7. **Train / validation / test split**:
   - Stratified split into train/test.
   - Secondary stratified split to obtain validation data.

8. **Missing‑value imputation**:
   - Computes the **median** of each feature on the training set.
   - Replaces `NaN` values in train/val/test with those medians.

9. **Feature scaling**:
   - Fits a `StandardScaler` on the imputed training data.
   - Applies the scaler to validation and test data.

10. **Class‑imbalance handling**:
    - Computes class weights with `compute_class_weight`.
    - Provides a dictionary of weights that can be used for training (optionally applied in the models).

---

## 5. Models

### 5.1. Neural network baseline (MLP)

The notebook defines and trains a **Multilayer Perceptron classifier** using TensorFlow / Keras:

- Input: standardized meta‑features.
- Architecture (typical configuration in the notebook):
  - Dense layer with 128 units, ReLU activation, followed by Dropout(0.2).
  - Dense layer with 64 units, ReLU activation, followed by Dropout(0.2).
  - Output dense layer with `num_classes` units and softmax activation.
- Loss: `sparse_categorical_crossentropy`.
- Optimizer: Adam with learning rate = 1e‑3.
- Metrics: accuracy.
- Early stopping:
  - Monitors validation accuracy.
  - Stops training when there is no improvement for a given number of epochs.
  - Restores the best weights.

After training, the notebook:

- Reports **best validation accuracy**.
- Plots training vs. validation accuracy per epoch.
- Evaluates on the test set:
  - Overall accuracy.
  - `classification_report` per class.
  - Confusion matrix.

The code also includes utilities for:

- Explaining a selected instance (printing key feature values and solver scores).
- Predicting the best solver for a specific row.
- Saving the trained model and preprocessing objects to disk as:
  - [`best_solver_mlp_keras.h5`](best_solver_mlp_keras.h5)
  - [`preproc_utils.npz`](preproc_utils.npz)

> Note: these files are saved in the runtime environment where you execute the notebook (e.g., Colab). They are not committed to this repository by default.

### 5.2. Reduced portfolio (without PSO)

In many datasets, PSO results may be missing or incomplete.
The notebook contains a variant of the pipeline that:

- Rebuilds labels considering only a subset of solvers (e.g., CPLEX, Gurobi, Gecode, GA).
- Drops the PSO column from the feature set.
- Repeats preprocessing, model training, and evaluation for this reduced portfolio.

The evaluation includes:

- Test accuracy.
- Classification report.
- Confusion matrix.
- **Top‑2 accuracy** (probability that the true solver is among the two most probable predictions).

### 5.3. XGBoost multi‑class classifier

The notebook also trains a **gradient‑boosted tree** model using XGBoost:

- Input: the same standardized features used for the neural network.
- Objective: `multi:softprob` (outputs a probability distribution over solvers).
- Handles class imbalance via explicit **sample weights** derived from class weights.
- Uses early stopping based on validation performance.

The evaluation part:

- Computes test accuracy.
- Prints a detailed classification report.
- Computes **Top‑2 accuracy**.
- Builds a confusion matrix.
- Extracts and plots **feature importances** (gain‑based) for the top features.

---

## 6. How to run the notebook

The easiest way is to use **Google Colab**.

1. Open the notebook:
   - Either click the Colab badge at the top of [`Baseline_JSP_Portafolio.ipynb`](Baseline_JSP_Portafolio.ipynb), or
   - Upload the notebook to your own Colab environment.

2. Install dependencies (first cell):
   - The notebook includes a `pip install` command to install:
     - `pandas`
     - `numpy`
     - `scikit-learn`
     - `tensorflow==2.15.*`
     - `matplotlib`
     - `xgboost` (installed later in the XGBoost section)

3. Upload your dataset:
   - Run the **data upload** cell.
   - Select your `.xlsx` file from your local machine.
   - Confirm the selected sheet (by default, the first sheet is used).
   - Inspect the printed columns and first rows to verify that the schema matches expectations.

4. Choose the label strategy:
   - Use the default cell if **lower metric values** mean better performance.
   - Switch to the alternative cell if **higher metric values** mean better performance.
   - Re‑run the preprocessing cells after changing this assumption.

5. Run the training cells:
   - Execute the MLP section to obtain a neural‑network baseline.
   - Optionally run the **reduced portfolio** section (without PSO).
   - Execute the **XGBoost** section for a tree‑based baseline.

6. Inspect the results:
   - Check terminal/log outputs for:
     - Validation and test accuracies.
     - Classification reports.
     - Confusion matrices.
     - Top‑k accuracies.
   - Look at the plots for training curves and feature importances.

7. Save artifacts (optional):
   - At the end of the MLP section, download:
     - [`best_solver_mlp_keras.h5`](best_solver_mlp_keras.h5)
     - [`preproc_utils.npz`](preproc_utils.npz)
   - These can be reused in other scripts or deployed as part of a solver‑selection service.

---

## 7. Environment and dependencies

Although the notebook includes `pip` cells, the core dependencies are:

- Python 3.9+ (Colab default is fine)
- `pandas`
- `numpy`
- `scikit-learn`
- `tensorflow==2.15.*`
- `matplotlib`
- `xgboost`
- `jupyter` (only if you run the notebook locally instead of Colab)

When running locally, you can create a virtual environment and install these packages using `pip` or `conda` before launching Jupyter.

---

## 8. Limitations and notes

- This is a **baseline** implementation intended for experimentation and teaching purposes:
  - Hyperparameters have not been extensively tuned.
  - No cross‑validation or automated model selection is implemented.
- The quality of predictions depends heavily on:
  - The representativeness and size of the dataset.
  - The correctness of the solver performance values.
  - The validity of the "lower is better" vs. "higher is better" assumption.
- Messages and comments within the notebook are partially in Spanish, but the workflow is straightforward:
  - Upload data → clean and inspect → build labels → preprocess → train models → evaluate → save.

---

## 9. How to cite / use

If you use this baseline in academic or industrial work, you can cite it generically as:

> Baseline JSP Solver Selection Portfolio — Neural Network and XGBoost baselines for job shop instance‑based algorithm selection.

Feel free to adapt the notebook, extend the feature set, add more solvers, or integrate advanced models (e.g., deeper networks, graph‑based models, or meta‑learning frameworks) on top of this baseline.