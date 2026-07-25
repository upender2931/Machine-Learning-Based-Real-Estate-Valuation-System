import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import shap

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Real Estate ML System", layout="wide")
st.title("🏠 Real Estate Valuation System")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv("real_estate.csv", sep="\t")
df.columns = df.columns.str.strip()

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
models = {
    "Linear Regression": pickle.load(open("linear_regression.pkl","rb")),
    "Decision Tree": pickle.load(open("decision_tree.pkl","rb")),
    "Random Forest": pickle.load(open("random_forest.pkl","rb")),
    "XGBoost": pickle.load(open("xgboost.pkl","rb"))
}

# --------------------------------------------------
# LOAD METRICS
# --------------------------------------------------
metrics = json.load(open("model_metrics.json"))
valid_models = [m for m in metrics if "Mean_R2" in metrics[m]]
best_model = max(valid_models, key=lambda x: metrics[x]["Mean_R2"])

# --------------------------------------------------
# LOAD TEST DATA
# --------------------------------------------------
X_test = pickle.load(open("X_test.pkl","rb"))
y_test = pickle.load(open("y_test.pkl","rb"))

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "step" not in st.session_state:
    st.session_state.step = 1
if "data" not in st.session_state:
    st.session_state.data = {}
if "predicted" not in st.session_state:
    st.session_state.predicted = False

data = st.session_state.data

# ==================================================
# STEP 1
# ==================================================
if st.session_state.step == 1:

    st.header("Step 1: Basic Property Details")

    data["State"] = st.selectbox("State", df["State"].unique(), help="Select state")
    data["City"] = st.selectbox("City", df["City"].unique(), help="Select city")

    property_type = st.selectbox("Property Type", df["Property_Type"].unique(), help="Type of property")
    data["Property_Type"] = property_type

    data["Facing"] = st.selectbox("Facing", df["Facing"].unique(), help="Direction")

    data["Area_sqft"] = st.number_input("Area (sqft)",100,10000, help="Total area")

    if property_type != "Open Plot":
        data["Furnishing_Type"] = st.selectbox("Furnishing", df["Furnishing_Type"].unique(), help="Furnishing level")
        data["Bedrooms"] = st.number_input("Bedrooms",0,10, help="Bedrooms")
        data["Bathrooms"] = st.number_input("Bathrooms",0,10, help="Bathrooms")
        data["Property_Age"] = st.number_input("Age",0,50, help="Property age")
        data["Floor_Number"] = st.number_input("Floor",0,100, help="Floor level")
        data["Total_Floors"] = st.number_input("Total Floors",1,100, help="Total floors")
        data["Two_Wheeler_Parking"] = st.number_input("2W Parking",0,10, help="Bike parking")
        data["Four_Wheeler_Parking"] = st.number_input("4W Parking",0,10, help="Car parking")
    else:
        data.update({"Furnishing_Type":"Unfurnished","Bedrooms":0,"Bathrooms":0,
                     "Property_Age":0,"Floor_Number":0,"Total_Floors":0,
                     "Two_Wheeler_Parking":0,"Four_Wheeler_Parking":0})

    if st.button("Next ➜"):
        st.session_state.step = 2

# ==================================================
# STEP 2
# ==================================================
elif st.session_state.step == 2:

    st.header("Step 2: Accessibility")

    data["Metro_Distance_km"] = st.number_input("Metro Distance",0.1, help="Distance to metro")
    data["Bus_Stop_Distance_km"] = st.number_input("Bus Distance",0.1, help="Distance to bus stop")
    data["School_Distance_km"] = st.number_input("School Distance",0.1, help="Distance to school")
    data["Hospital_Distance_km"] = st.number_input("Hospital Distance",0.1, help="Distance to hospital")
    data["Mall_Distance_km"] = st.number_input("Mall Distance",0.1, help="Distance to mall")

    data["Traffic_Index"] = st.slider("Traffic",0.0,100.0, help="Traffic level")
    data["Walkability_Score"] = st.slider("Walkability",0.0,100.0, help="Walk score")
    data["Commute_Time_Minutes"] = st.number_input("Commute Time",1, help="Travel time")

    if st.button("⬅ Back"):
        st.session_state.step = 1
    if st.button("Next ➜"):
        st.session_state.step = 3

# ==================================================
# STEP 3
# ==================================================
elif st.session_state.step == 3:

    st.header("Step 3: Environment")

    data["Flood_Risk"] = st.slider("Flood Risk",0.0,100.0, help="Flood probability")
    data["Air_Quality_Index"] = st.slider("AQI",0.0,300.0, help="Air quality")
    data["Noise_Index"] = st.slider("Noise",0.0,120.0, help="Noise level")
    data["Heat_Risk"] = st.slider("Heat Risk",0.0,100.0, help="Heat exposure")
    data["Seismic_Risk"] = st.slider("Seismic Risk",0.0,5.0, help="Earthquake risk")
    data["Crime_Rate"] = st.slider("Crime Rate",0.0,100.0, help="Crime level")

    data["Rental_Yield_%"] = st.slider("Rental Yield",0.0,10.0, help="Rental return")
    data["Historical_Price_Appreciation_5Y_%"] = st.slider("Growth",0.0,50.0, help="5 year growth")

    if st.button("⬅ Back"):
        st.session_state.step = 2
    if st.button("Next ➜"):
        st.session_state.step = 4

# ==================================================
# STEP 4 (IMPORTANT)
# ==================================================
elif st.session_state.step == 4:

    st.header("Prediction")

    if st.button("Predict Price"):
        st.session_state.predicted = True

    if st.session_state.predicted:

        # ---------------------------
        # PREDICTION
        # ---------------------------
        model_prices = {}
        for name, model in models.items():
            pred = model.predict(pd.DataFrame([data]))[0]
            model_prices[name] = np.expm1(pred)

        st.success(f"🏆 Best Model: {best_model}")
        st.success(f"💰 Price: ₹ {model_prices[best_model]:,.0f}")

        # ---------------------------
        # COMMON DATA
        # ---------------------------
        best_model_obj = models[best_model]
        y_pred = np.expm1(best_model_obj.predict(X_test))
        y_actual = np.expm1(y_test)
        residuals = y_actual - y_pred

        # =============================
        # 1. MODEL COMPARISON
        # =============================
        st.subheader("Model Comparison")
        fig1, ax1 = plt.subplots()
        bars = ax1.bar(model_prices.keys(), model_prices.values())
        for bar in bars:
            ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                     f"₹{bar.get_height():,.0f}", ha='center')
        st.pyplot(fig1)

        # =============================
        # 2. ACTUAL VS PREDICTED
        # =============================
        st.subheader("Actual vs Predicted")
        fig2, ax2 = plt.subplots()
        ax2.scatter(y_actual,y_pred)
        ax2.plot([min(y_actual),max(y_actual)],
                 [min(y_actual),max(y_actual)],"r--")
        st.pyplot(fig2)

        # =============================
        # 3. RESIDUAL
        # =============================
        st.subheader("Residual Plot")
        fig3, ax3 = plt.subplots()
        ax3.scatter(y_pred,residuals)
        ax3.axhline(0,color="red")
        st.pyplot(fig3)

        # =============================
        # 4. CORRELATION (CLEAR)
        # =============================
        st.subheader("Correlation Matrix")
        try:
            import seaborn as sns
            fig4, ax4 = plt.subplots(figsize=(12,8))
            sns.heatmap(df.corr(numeric_only=True),
                        annot=True, fmt=".2f",
                        cmap="coolwarm", ax=ax4)
            plt.xticks(rotation=45)
            st.pyplot(fig4)
        except:
            st.warning("Install seaborn")

        # =============================
        # 5. FEATURE IMPORTANCE
        # =============================
        st.subheader("Feature Importance")
        try:
            tree = best_model_obj.named_steps["model"]
            imp = tree.feature_importances_
            names = best_model_obj.named_steps["preprocessor"].get_feature_names_out()

            df_imp = pd.DataFrame({"Feature":names,"Importance":imp}).sort_values(by="Importance",ascending=False).head(10)

            fig5, ax5 = plt.subplots()
            ax5.barh(df_imp["Feature"],df_imp["Importance"],color="skyblue")
            ax5.invert_yaxis()
            st.pyplot(fig5)
        except:
            st.info("Not available")

        # =============================
        # 6. SHAP
        # =============================
        st.subheader("SHAP")
        try:
            Xp = best_model_obj.named_steps["preprocessor"].transform(X_test)
            explainer = shap.TreeExplainer(best_model_obj.named_steps["model"])
            shap_values = explainer.shap_values(Xp)
            fig6 = plt.figure()
            shap.summary_plot(shap_values,Xp,show=False)
            st.pyplot(fig6)
        except:
            st.info("SHAP not supported")

        if st.button("Restart"):
            st.session_state.step = 1
            st.session_state.predicted = False
            st.session_state.data = {}
