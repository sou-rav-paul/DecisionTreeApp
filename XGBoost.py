import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Set up the page configuration
st.set_page_config(page_title="Economic Expenditure Predictor", page_icon="📊", layout="wide")

st.title("📊 XGBoost Economic Expenditure Predictor")
st.write("Predict 6 government spending sectors (as % of GDP) based on historical economic data.")

TARGET_COLS = [
    "health_expense",
    "total_gov_expense",
    "education_expense",
    "capital_formation_exp",
    "military_expense",
    "r&d_expense"
]

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("refined_economic_dataset.csv")
        return df
    except FileNotFoundError:
        st.error("Dataset not found! Please ensure 'refined_economic_dataset.csv' is uploaded.")
        return None

@st.cache_resource
def load_model_assets():
    """Loads the pre-trained XGBoost model and country mapping dictionaries."""
    try:
        model = joblib.load("xgboost_budget_model.pkl")
        country_mapping = joblib.load("country_mapping.pkl")
        return model, country_mapping
    except FileNotFoundError:
        st.error("Model files not found! Please ensure the .pkl files are uploaded.")
        return None, None

# Load data and pre-trained model
df = load_data()
model, country_mapping = load_model_assets()

if df is not None and model is not None:
    valid_countries = sorted(df["country_name"].unique())
    
    # Sidebar for User Inputs
    st.sidebar.header("Input Parameters")
    user_country = st.sidebar.selectbox("Select Country", valid_countries)
    user_year = st.sidebar.number_input("Enter Year", min_value=2000, max_value=2100, value=2024, step=1)
    
    # Dynamic GDP hint based on historical data
    hist_gdp_max = df[df['country_name'] == user_country]['total_gdp'].max()
    hist_gdp_min = df[df['country_name'] == user_country]['total_gdp'].min()
    st.sidebar.caption(f"Historical GDP Range: \n${hist_gdp_min:,.0f} - \n${hist_gdp_max:,.0f}")
    
    user_gdp = st.sidebar.number_input("Enter Total GDP (USD)", min_value=1.0, value=float(hist_gdp_max), step=1e10, format="%.2f")

    if st.sidebar.button("Predict Expenditure"):
        # Look for actual historical data to compare
        actual_exists = False
        last_year_percentages = None
        
        if user_year <= 2023:
            past_data = df[(df['country_name'] == user_country) & (df['year'] == user_year)]
            if not past_data.empty:
                actual_exists = True
                actual_percentages = past_data[TARGET_COLS].values[0]
        else:
            past_data = df[(df['country_name'] == user_country) & (df['year'] == df['year'].max())]
            if not past_data.empty:
                last_year_percentages = past_data[TARGET_COLS].values[0]

        # Formatting inputs for prediction
        input_df = pd.DataFrame({
            "country_encoded": [country_mapping[user_country]],
            "year": [user_year],
            "log_gdp": [np.log10(user_gdp)]
        })
        
        # Predicting
        pred_percentages = model.predict(input_df)[0]
        billion_usd = (pred_percentages / 100 * user_gdp) / 1_000_000_000
        
        # Build Results Table
        if actual_exists:
            results = pd.DataFrame({
                "Sector": [c.replace("_", " ").title() for c in TARGET_COLS],
                "Actual (% GDP)": np.round(actual_percentages, 2),
                "Predicted (% GDP)": np.round(pred_percentages, 2),
                "Predicted (Billion USD)": np.round(billion_usd, 2)
            })
            chart_title = f"Actual vs Predicted Expenditure - {user_country} ({user_year})"
            compare_vals = actual_percentages
            compare_label = 'Actual Data'
        else:
            ref = last_year_percentages if last_year_percentages is not None else [np.nan]*6
            results = pd.DataFrame({
                "Sector": [c.replace("_", " ").title() for c in TARGET_COLS],
                "Last Known (% GDP)": np.round(ref, 2),
                "Predicted (% GDP)": np.round(pred_percentages, 2),
                "Predicted (Billion USD)": np.round(billion_usd, 2)
            })
            chart_title = f"Last Known vs Predicted Expenditure - {user_country} ({user_year})"
            compare_vals = ref if last_year_percentages is not None else [0]*6
            compare_label = 'Last Known Data'

        # Display Metrics & Table
        st.subheader(f"Prediction Results for {user_country} ({user_year})")
        st.dataframe(results, use_container_width=True, hide_index=True)
        
        # Display Chart
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(TARGET_COLS))
        width = 0.35
        
        ax.bar(x - width/2, compare_vals, width=width, label=compare_label, color='salmon')
        ax.bar(x + width/2, results["Predicted (% GDP)"], width=width, label='Predicted Data', color='skyblue')
        
        ax.set_title(chart_title, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", " ").title() for c in TARGET_COLS], rotation=15)
        ax.set_ylabel("% of GDP")
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        st.pyplot(fig)
