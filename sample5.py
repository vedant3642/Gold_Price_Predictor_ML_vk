import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from lightgbm import LGBMRegressor
from datetime import datetime, timedelta

# Load dataset
gold_data = pd.read_csv('Daily.csv')

# Convert 'Date' column to datetime format
gold_data["Date"] = pd.to_datetime(gold_data["Date"])

# Filter data for the last 5 years
end_date = gold_data["Date"].max()
start_date = end_date - timedelta(days=5*365)  # Approximate 5 years
gold_data = gold_data[gold_data["Date"] >= start_date]

# Sort by date
gold_data.sort_values(by="Date", inplace=True)

# Convert price columns to numeric, handling commas
for col in gold_data.columns[1:]:  # Exclude 'Date' column
    gold_data[col] = gold_data[col].astype(str).str.replace(",", "").astype(float)

# Drop rows with missing values
gold_data.dropna(inplace=True)

# Feature Engineering - Use only date-based features
gold_data["DateNumeric"] = (gold_data["Date"] - gold_data["Date"].min()).dt.days
gold_data["Day"] = gold_data["Date"].dt.day
gold_data["Month"] = gold_data["Date"].dt.month
gold_data["Year"] = gold_data["Date"].dt.year
gold_data["DayOfWeek"] = gold_data["Date"].dt.weekday

# Define features and target variable
date_features = ["DateNumeric", "Day", "Month", "Year", "DayOfWeek"]
X = gold_data[date_features]
y = gold_data["USD"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize regressors
models = {
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=30),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.01, max_depth=15),
    "Decision Tree": DecisionTreeRegressor(max_depth=20),
    "LightGBM": LGBMRegressor(n_estimators=200, learning_rate=0.01, max_depth=15)
}

# Train models and store R² scores
model_scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model_scores[name] = metrics.r2_score(y_test, y_pred)

# Fit a Linear Regression model on DateNumeric only
lr = LinearRegression()
lr.fit(X_train[["DateNumeric"]], y_train)

# Streamlit app setup
st.set_page_config(page_title="Gold Price Prediction", layout="wide")
st.title("📈 Gold Price Prediction Web App")
st.write("Predict future gold prices using advanced ML models.")

# User selects model
st.sidebar.header("🔍 Select Model")
model_choice = st.sidebar.selectbox("Choose a model:", list(models.keys()))
selected_model = models[model_choice]

# User inputs future date
st.sidebar.header("📅 Select Future Date")
future_date = st.sidebar.date_input("Choose a date:", min_value=datetime.today())
future_numeric = (pd.to_datetime(future_date) - gold_data["Date"].min()).days
future_day = future_date.day
future_month = future_date.month
future_year = future_date.year
future_dayofweek = future_date.weekday()

# Prepare future input based on date features only
future_X = pd.DataFrame([{
    "DateNumeric": future_numeric,
    "Day": future_day,
    "Month": future_month,
    "Year": future_year,
    "DayOfWeek": future_dayofweek
}])

# Predict future price
future_price = selected_model.predict(future_X)[0]

# Apply Linear Trend Adjustment to ensure price increases realistically
trend_adjustment = lr.predict([[future_numeric]])[0]
adjusted_future_price = max(future_price, trend_adjustment)

# Get model accuracy
accuracy = model_scores[model_choice]

# Display results
st.subheader("📊 Prediction Results")
st.markdown(
    f"""
    <div style="padding: 15px; border-radius: 10px; background-color: #f9f9f9; text-align: center;">
        <h2 style="color: #2E86C1;">Predicted Gold Price on {future_date}: <strong>${adjusted_future_price:.2f}</strong></h2>
        <h4>Model Accuracy (R² Score): <strong>{accuracy:.4f}</strong></h4>
    </div>
    """,
    unsafe_allow_html=True
)

# Visualization
st.write("## 📌 Gold Price Trend Over Time")
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(x=gold_data["Date"], y=gold_data["USD"], label="Historical Prices", color='blue')
ax.scatter(future_date, adjusted_future_price, color='red', label="Predicted Price", marker='o', s=150)
ax.set_xlabel("Date")
ax.set_ylabel("Gold Price (USD)")
ax.set_title("Gold Price Prediction Trend")
ax.legend()
st.pyplot(fig)

st.sidebar.markdown("---")
st.sidebar.info("📌 This app provides machine learning-based predictions for future gold prices using multiple models. Choose a model and a future date to get predictions!")
