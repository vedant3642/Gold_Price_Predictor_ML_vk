import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from lightgbm import LGBMRegressor
from datetime import datetime

# Load dataset
gold_data = pd.read_csv('Daily.csv')

# Convert 'Date' column to datetime format
gold_data["Date"] = pd.to_datetime(gold_data["Date"])

# Convert 'Date' to a numerical value (days since the first date)
gold_data["DateNumeric"] = (gold_data["Date"] - gold_data["Date"].min()).dt.days

# Convert price columns to numeric, handling commas
for col in gold_data.columns[1:]:  # Exclude 'Date' column
    gold_data[col] = gold_data[col].astype(str).str.replace(",", "").astype(float)

# Drop rows with missing values
gold_data.dropna(inplace=True)

# Sort by date
gold_data.sort_values(by="Date", inplace=True)

# Define features and target variable
X = gold_data.drop(columns=["USD", "Date"])  # Include DateNumeric, exclude original 'Date'
y = gold_data["USD"]  # Target: Gold price in USD

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize regressors
regressor1 = RandomForestRegressor(n_estimators=200)
regressor2 = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05)
regressor3 = DecisionTreeRegressor(max_depth=10)
regressor4 = LGBMRegressor(n_estimators=200)

# Train models
regressor1.fit(X_train, y_train)
regressor2.fit(X_train, y_train)
regressor3.fit(X_train, y_train)
regressor4.fit(X_train, y_train)

# Predict on test data
test_data_prediction1 = regressor1.predict(X_test)
test_data_prediction2 = regressor2.predict(X_test)
test_data_prediction3 = regressor3.predict(X_test)
test_data_prediction4 = regressor4.predict(X_test)

# Calculate R² scores
error_score1 = metrics.r2_score(y_test, test_data_prediction1)
print("R squared error for Random Forest Regressor: ", error_score1)
error_score2 = metrics.r2_score(y_test, test_data_prediction2)
print("R squared error for Gradient Boosting Regressor: ", error_score2)
error_score3 = metrics.r2_score(y_test, test_data_prediction3)
print("R squared error for Decision Tree Regressor: ", error_score3)
error_score4 = metrics.r2_score(y_test, test_data_prediction4)
print("R squared error for LightGBM Regressor: ", error_score4)

# take this ML model for gold price prediction , generate a professional web app using  streamlit 
#  where app gives prediction of gold price of each model with its respective accuracy when user inputs 
# future dates. User can chose model out of four mentioned and will get price of future dates with accuracy. \
# Price of gold should be accurate that is increasing linearly with dates. Take your time but write code without errors .