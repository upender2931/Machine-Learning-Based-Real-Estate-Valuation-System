import pandas as pd
import numpy as np
import pickle
import json

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import ttest_rel
import scipy.stats as stats

print("Loading dataset...")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
df = pd.read_csv("real_estate.csv", sep="\t")
df.columns = df.columns.str.strip()

# -------------------------------------------------
# ✅ PRICE FIX (Lakh / Crore)
# -------------------------------------------------
def convert_price(x):
    x = str(x).lower().strip()
    try:
        if "cr" in x:
            return float(x.replace("cr", "").strip()) * 10000000
        elif "l" in x:
            return float(x.replace("l", "").strip()) * 100000
        else:
            return float(x)
    except:
        return np.nan

df["Price"] = df["Price"].apply(convert_price)
df = df.dropna(subset=["Price"])

print("Final dataset size:", df.shape)

# -------------------------------------------------
# FEATURES
# -------------------------------------------------
X = df.drop("Price", axis=1)
y = np.log1p(df["Price"])

# -------------------------------------------------
# PREPROCESS
# -------------------------------------------------
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numerical_features = X.select_dtypes(exclude=["object"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

# -------------------------------------------------
# SPLIT
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

metrics_dict = {}

# -------------------------------------------------
# ✅ EVALUATION FUNCTION (FIXED)
# -------------------------------------------------
def evaluate_model(name, model):
    y_pred_log = model.predict(X_test)

    y_actual = np.expm1(y_test)
    y_pred = np.expm1(y_pred_log)

    mae = mean_absolute_error(y_actual, y_pred)
    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_actual, y_pred)

    print("\n" + "="*40)
    print(name)
    print("="*40)
    print(f"MAE  : {mae:,.2f}")
    print(f"MSE  : {mse:,.2f}")
    print(f"RMSE : {rmse:,.2f}")
    print(f"R2   : {r2:.4f}")
    print("="*40)

# -------------------------------------------------
# LINEAR REGRESSION
# -------------------------------------------------
lr = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

lr.fit(X_train, y_train)
evaluate_model("Linear Regression", lr)

lr_scores = cross_val_score(lr, X, y, cv=kf, scoring="r2")
metrics_dict["Linear Regression"] = {"Mean_R2": np.mean(lr_scores)}

# -------------------------------------------------
# DECISION TREE
# -------------------------------------------------
dt = Pipeline([
    ("preprocessor", preprocessor),
    ("model", DecisionTreeRegressor(max_depth=12))
])

dt.fit(X_train, y_train)
evaluate_model("Decision Tree", dt)

dt_scores = cross_val_score(dt, X, y, cv=kf, scoring="r2")
metrics_dict["Decision Tree"] = {"Mean_R2": np.mean(dt_scores)}

# -------------------------------------------------
# RANDOM FOREST
# -------------------------------------------------
rf = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=80, max_depth=12, n_jobs=-1))
])

rf.fit(X_train, y_train)
evaluate_model("Random Forest", rf)

rf_scores = cross_val_score(rf, X, y, cv=kf, scoring="r2")
metrics_dict["Random Forest"] = {"Mean_R2": np.mean(rf_scores)}

# -------------------------------------------------
# XGBOOST
# -------------------------------------------------
print("\nTraining XGBoost...")

xgb = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1500,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1
    ))
])

xgb.fit(X_train, y_train)
evaluate_model("XGBoost", xgb)

xgb_scores = cross_val_score(xgb, X, y, cv=kf, scoring="r2")
metrics_dict["XGBoost"] = {"Mean_R2": np.mean(xgb_scores)}

# -------------------------------------------------
# PRINT FINAL
# -------------------------------------------------
print("\nFinal Mean R2 Scores:")
for k, v in metrics_dict.items():
    print(k, ":", round(v["Mean_R2"], 6))

print("\n✅ DONE SUCCESSFULLY")
