import pickle
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("\n==============================")
print("   MODEL EVALUATION RESULTS")
print("==============================\n")

# -------------------------------------------------
# LOAD TEST DATA
# -------------------------------------------------
X_test = pickle.load(open("X_test.pkl", "rb"))
y_test = pickle.load(open("y_test.pkl", "rb"))

# -------------------------------------------------
# LOAD TRAINED PIPELINE MODELS
# -------------------------------------------------
models = {
    "Linear Regression": pickle.load(open("linear_regression.pkl", "rb")),
    "Decision Tree": pickle.load(open("decision_tree.pkl", "rb")),
    "Random Forest": pickle.load(open("random_forest.pkl", "rb")),
    "XGBoost": pickle.load(open("xgboost.pkl", "rb"))   # GridSearch best model
}

# -------------------------------------------------
# EVALUATION LOOP
# -------------------------------------------------
for name, model in models.items():

    # Predict (pipeline handles preprocessing)
    y_pred_log = model.predict(X_test)

    # Reverse log transformation
    y_actual = np.expm1(y_test)
    y_pred = np.expm1(y_pred_log)

    # Metrics
    mae = mean_absolute_error(y_actual, y_pred)
    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_actual, y_pred)

    print(f"{name}")
    print(f"MAE  : {mae:,.2f}")
    print(f"MSE  : {mse:,.2f}")
    print(f"RMSE : {rmse:,.2f}")
    print(f"R2   : {r2:.4f}")
    print("-" * 50)

print("\n✅ Evaluation Completed Successfully\n")
