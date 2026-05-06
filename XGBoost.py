import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# --- 1. Page Configuration ---
st.set_page_config(page_title="Budget Predictor", layout="wide")
st.title("📊 Government Budget Expenditure Predictor (XGBoost)")

# --- 2. Load Data and Train Model ---
# 🚨 CRITICAL FIX: We MUST use @st.cache_resource. 
# @st.cache_data will crash because XGBoost models cannot be hashed the same way as standard Python objects.
@st.cache_resource
def load_and_train_xgb():
    df = pd.read_csv("refined_economic_dataset.csv")
    
    TARGET_COLS = ["health_expense", "total_gov_expense", "education_expense",
                   "capital_formation_exp", "military_expense", "r&d_expense"]
    
    # Preprocessing
    le = LabelEncoder()
    df["country_encoded"] = le.fit_transform(df["country_name"])
    country_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    df["log_gdp"] = np.log10(df["total_gdp"])
    
    FEATURE_COLS = ["country_encoded", "year", "log_gdp"]
    X = df[FEATURE_COLS]
    Y = df[TARGET_COLS]
    
    # Train XGBoost Model
    base_xgb = xgb.XGBRegressor(
        n_estimators=500,     # Lowered to 500 to prevent Cloud Memory Crashes
        max_depth=4,
        learning_rate=0.05,
        reg_alpha=1,
        subsample=0.7,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=2              # Safe setting for cloud environments (prevents crashes)
    )
    model = MultiOutputRegressor(base_xgb)
    model.fit(X, Y)
    
    return df, model, le, country_mapping, TARGET_COLS

try:
    df, model, le, country_mapping, TARGET_COLS = load_and_train_xgb()
except Exception as e:
    st.error(f"Error loading data: {e}. Please ensure 'refined_economic_dataset.csv' is uploaded to GitHub.")
    st.stop()

VALID_COUNTRIES = sorted(df["country_name"].unique())

# --- 3. Streamlit Sidebar (User Inputs) ---
st.sidebar.header("Input Parameters")
user_country = st.sidebar.selectbox("Select Country", VALID_COUNTRIES)
user_year = st.sidebar.number_input("Enter Year", min_value=2000, max_value=2100, value=2025, step=1)

# Dynamic GDP input placeholder based on user selection
if user_year < 2024:
    past_data1 = df[(df['country_name'] == user_country) & (df['year'] == user_year)]
    if not past_data1.empty:
        default_gdp = past_data1['total_gdp'].values[0]
    else:
        default_gdp = df[df['country_name'] == user_country]['total_gdp'].mean()
else:
    max_gdp = df[df['country_name'] == user_country]['total_gdp'].max()
    min_gdp = df[df['country_name'] == user_country]['total_gdp'].min()
    default_gdp = max_gdp  # Setting a sensible default
    st.sidebar.caption(f"Historical GDP Range: ${min_gdp:,.0f} to ${max_gdp:,.0f}")

user_gdp = st.sidebar.number_input("Enter Total GDP (USD)", value=float(default_gdp), step=1000000000.0)

# --- 4. Prediction Button and Logic ---
if st.sidebar.button("Predict Expenditure"):
    
    country_code = country_mapping[user_country]
    log_gdp = np.log10(user_gdp)
    
    input_df = pd.DataFrame({
        "country_encoded": [country_code],
        "year": [user_year],
        "log_gdp": [log_gdp]
    })
    
    pred_percentages = model.predict(input_df)[0]
    billion_usd = (pred_percentages / 100 * user_gdp) / 1_000_000_000
    
    # Check if actual historical data exists
    actual_exists = False
    if user_year <= 2023:
        past_data = df[(df['country_name'] == user_country) & (df['year'] == user_year)]
        if not past_data.empty:
            actual_exists = True
            actual_percentages = past_data[TARGET_COLS].values[0]
            
    if not actual_exists:
        # Get the latest available year for comparison
        latest_year = df[df['country_name'] == user_country]['year'].max()
        past_data = df[(df['country_name'] == user_country) & (df['year'] == latest_year)]
        last_year_percentages = past_data[TARGET_COLS].values[0]

    # Format the sector names for display
    display_sectors = [c.replace("_", " ").title() for c in TARGET_COLS]

    # Create the results DataFrame
    if actual_exists:
        results_df = pd.DataFrame({
            "Sector": display_sectors,
            "Actual (%)": np.round(actual_percentages, 2),
            "Predicted (%)": np.round(pred_percentages, 2),
            "Predicted (Billion USD)": np.round(billion_usd, 2)
        })
    else:
        results_df = pd.DataFrame({
            "Sector": display_sectors,
            "Nearest Past Data (%)": np.round(last_year_percentages, 2),
            "Predicted (%)": np.round(pred_percentages, 2),
            "Predicted (Billion USD)": np.round(billion_usd, 2)
        })

    # --- 5. Display Results in Streamlit ---
    st.subheader(f"Results for {user_country} ({user_year})")
    
    # Display Table
    st.dataframe(results_df, use_container_width=True)
    
    # Display Chart
    st.subheader("Comparison Chart")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(TARGET_COLS))
    width = 0.35
    
    if actual_exists:
        ax.bar(x - width/2, actual_percentages, width=width, label='Actual Data', color='salmon')
        ax.set_title(f"Actual vs Predicted Expenditure for {user_country} ({user_year})")
    else:
        ax.bar(x - width/2, last_year_percentages, width=width, label=f'Last Available Data ({latest_year})', color='salmon')
        ax.set_title(f"Latest Available vs Predicted Expenditure for {user_country} ({user_year})")
        
    ax.bar(x + width/2, results_df["Predicted (%)"], width=width, label='Predicted Data (XGBoost)', color='skyblue')
    
    ax.set_xticks(x)
    ax.set_xticklabels(display_sectors, rotation=15)
    ax.set_ylabel("% of GDP")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    st.pyplot(fig)

else:
    st.info("👈 Enter the parameters in the sidebar and click 'Predict Expenditure' to view the results.")
