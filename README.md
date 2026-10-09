# ML Experiment Tracking with MLflow — California Housing

## 1. Task Description

This project uses **MLflow** to track and compare machine learning experiments for a **House Price Prediction** problem.

The same regression model (`GradientBoostingRegressor`) is trained **three times** on the **California Housing** dataset, each time with different hyperparameters. For every run, MLflow records:

- **Parameters:** `max_depth`, `learning_rate`
- **Metrics:** RMSE, MAE, R²
- **Model artifact:** the trained model
- **Experiment results:** a comparison of all runs in the MLflow UI

Finally, the best-performing model is selected using **RMSE** as the primary metric.

**Workflow:** Train → Track → Compare → Select

### Dataset

- **Source:** California Housing dataset (`sklearn.datasets.fetch_california_housing`), downloaded automatically on first run.
- **Size:** 20,640 samples, 8 features.
- **Target:** `MedHouseVal` (median house value).
- **Split:** 80% training / 20% validation (`random_state=42` for reproducibility).

### Experiment Configurations

| Run   | Max Depth | Learning Rate |
|-------|-----------|---------------|
| Run 1 | 3         | 0.1           |
| Run 2 | 5         | 0.05          |
| Run 3 | 7         | 0.01          |

All other hyperparameters (e.g. `n_estimators`) are kept at their default values so the comparison is fair.

---

## 2. Project Structure

```
mlflow-task/
│
├── train.py
├── requirements.txt
└── README.md
```

---

## 3. How to Run the Project

### 3.1 Clone the repository

```bash
git clone https://github.com/AhmedSayed-21/MLflow-task.git
cd mlflow-task
```

### 3.2 Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3.3 Install the dependencies

```bash
pip install -r requirements.txt
```

### 3.4 Train the models

```bash
python train.py
```

This will:

1. Load the California Housing dataset.
2. Split it into 80% training and 20% validation.
3. Train the three experiments.
4. Log parameters, metrics, and the trained model of each run to MLflow.
5. Print the best run (lowest RMSE).

Results are stored locally in `mlflow.db` (SQLite).

### 3.5 Open the MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open **http://127.0.0.1:5000** in your browser, select the experiment **California Housing - House Price Prediction**, tick the three runs, and click **Compare**.

> **Note:** Run the command from the same folder that contains `mlflow.db`.

---

## 4. Comparison Table

Evaluated on the 20% validation set.

| Run   | Max Depth | Learning Rate | RMSE   | MAE    | R²     |
|-------|-----------|---------------|--------|--------|--------|
| Run 1 | 3         | 0.1           | 0.5422 | 0.3716 | 0.7756 |
| Run 2 | 5         | 0.05          | 0.5198 | 0.3531 | 0.7938 |
| Run 3 | 7         | 0.01          | 0.6973 | 0.5311 | 0.6290 |

> Lower RMSE and MAE are better. A higher R² is better.

---

## 5. Selected Best Model

**Best model: Run 2** — `max_depth = 5`, `learning_rate = 0.05` (RMSE = 0.5198).

It was also registered in the MLflow Model Registry as `california-housing-best-model`.

---

## 6. Why This Model Was Selected

RMSE is the primary selection metric for this task, and **Run 2 achieved the lowest RMSE (0.5198)** of the three runs, meaning its predictions on the validation set are the closest to the true house prices. It also had the best MAE (0.3531) and the best R² (0.7938), so the choice is consistent across all three metrics.

- **Run 1** (`max_depth = 3`, `learning_rate = 0.1`, RMSE = 0.5422) uses shallow trees, so it has limited capacity to capture the more complex relationships in the data.
- **Run 3** (`max_depth = 7`, `learning_rate = 0.01`, RMSE = 0.6973) uses deep trees but a very small learning rate. With the default number of trees, the model learns too slowly and does not converge enough.
- **Run 2** (`max_depth = 5`, `learning_rate = 0.05`) offers the best balance between model complexity and learning speed.

This also shows the trade-off between `max_depth` and `learning_rate`: both must be tuned together, and a lower learning rate normally needs more trees to perform well.

---

## 7. Technologies Used

- Python
- MLflow
- scikit-learn
- pandas
- NumPy
