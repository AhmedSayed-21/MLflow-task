import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RANDOM_SEED = 42
EXPERIMENT_NAME = "California Housing - House Price Prediction"

# ---------- Part 1: Prepare the dataset ----------
data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)

# ---------- Part 2: The three configurations ----------
configs = [
    {"run_name": "Run 1", "max_depth": 3, "learning_rate": 0.1},
    {"run_name": "Run 2", "max_depth": 5, "learning_rate": 0.05},
    {"run_name": "Run 3", "max_depth": 7, "learning_rate": 0.01},
]

mlflow.set_experiment(EXPERIMENT_NAME)

# ---------- Part 3: Train + track + log model ----------
for cfg in configs:
    with mlflow.start_run(run_name=cfg["run_name"]):
        # 1) Train the model
        model = GradientBoostingRegressor(
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            random_state=RANDOM_SEED,
        )
        model.fit(X_train, y_train)

        # 2) Calculate validation metrics
        preds = model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        mae = float(mean_absolute_error(y_val, preds))
        r2 = float(r2_score(y_val, preds))

        # 3) Log parameters
        mlflow.log_param("max_depth", cfg["max_depth"])
        mlflow.log_param("learning_rate", cfg["learning_rate"])

        # 4) Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # 5) Save the trained model as an MLflow artifact
        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        print(
            f"{cfg['run_name']} | max_depth={cfg['max_depth']} "
            f"lr={cfg['learning_rate']} | "
            f"RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}"
        )

# ---------- Part 4: Pick the best run (lowest RMSE) ----------
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.rmse ASC"],
)
best = runs.iloc[0]
print("\nBest run:", best["tags.mlflow.runName"])
print("RMSE:", round(best["metrics.rmse"], 4))