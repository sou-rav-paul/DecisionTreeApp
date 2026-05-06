
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb

# Set up the page configuration
st.set_page_config(page_title="Economic Expenditure Predictor", page_icon="📊", layout="wide")

st.title("📊 XGBoost Economic Expenditure Predictor")
st.write("Predict 6 government spending sectors (as % of GDP) based on historical economic data.")

# Target and Feature column definitions
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
    # Ensure "refined_economic_dataset.csv" is in the same directory
    try:
        df = pd.read_csv("refined_economic_dataset.csv")
        return df
    except FileNotFoundError:
        st.error("Dataset not found! Please ensure 'refined_economic_dataset.csv' is uploaded to the repository.")
        return None

@st.cache_resource
def train_model(df):
    """Encodes data and trains the XGBoost model using the best found hyperparameters."""
    le = LabelEncoder()
    df["country_encoded"] = le.fit_transform(df["country_name"])
    country_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    
    df["log_gdp"] = np.log10(df["total_gdp"])
    
    FEATURE_COLS = ["country_encoded", "year", "log_gdp"]
    X = df[FEATURE_COLS]
    Y = df[TARGET_COLS]
    
    # Best Parameters from GridSearch
    base_xgb = xgb.XGBRegressor(
        objective='reg:squarederror', 
        random_state=42, 
        n_estimators=1000,
        max_depth=4,
        learning_rate=0.05,
        reg_alpha=1,
        subsample=0.7,
        
    )
    
    model = MultiOutputRegressor(base_xgb)
    model.fit(X, Y)
    
    return model, country_mapping, df

# Load and prepare data/model
df = load_data()

if df is not None:
    model, country_mapping, df = train_model(df)
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
