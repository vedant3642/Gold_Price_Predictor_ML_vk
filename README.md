#  Gold Price Predictor – ML Project
This project is a machine learning-based Gold Price Predictor that allows users to forecast the future price of gold. Users can interactively choose from four different regression models to make predictions based on historical gold price data.

## 🔍 Features
📊 Model Selection: Choose from the following regression models:
Random Forest Regressor
Gradient Boosting Regressor
LightGBM Regressor
Decision Tree Regressor

📅 Future Date Prediction: Input any future date to get the predicted gold price for that specific day.

📈 Accurate Forecasting: Trained on real historical gold prices with preprocessing and time series features.

🖥️ Interactive Interface: Clean and user-friendly UI built for easy access and experimentation built with streamlit implementation.

## ⚙️ How It Works
The user selects a regression model.
Inputs a future date.
The app returns the predicted price of gold in USD per ounce of gold on that date using the selected model.

## 📁 File Information
MLmodel.py : 4 gold price predictor Models code.
sample.py : Implementation of 4 models to predict gold price.
sample5.py : Implementation of 4 models to predict gold price and here last 5 years data used from dataset to increase accuracy.
sample10.py : Implementation of 4 models to predict gold price and here last 10 years data used from dataset to increase accuracy.

## ✅ Conclusion
The dataset used for this project spans from 1978 to 2023, but after data cleaning, the effective range is from 1993 ($380) to 2023 ($1960) — covering 30 years of historical gold prices. Over this period, the price of gold has increased in a mostly linear fashion, rising approximately $1600, which aligns well with the assumptions made by the regression models.

However, a sharp and unprecedented rise in gold prices occurred from 2023 to 2025, jumping from $1960 to $3300 — a $1300 increase in just two years. This sudden spike is outside the scope of the training data, which ends in 2023. As a result, the models struggle to accurately predict prices beyond 2023, particularly under such volatile market conditions.

In conclusion, while the Random Forest, Gradient Boosting, LightGBM, and Decision Tree Regressors perform well on historical data with a linear trend, their predictions for recent years are limited by the lack of updated data. To improve accuracy, especially during periods of sudden market changes, the models must be retrained on the most recent data to capture these new trends.
