import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Load your sequential time series data
# Assuming your data is in a CSV file
data = pd.read_csv(r"C:\Users\konda\OneDrive\Desktop\Final_project\input data\NSW\input aggregated hourly prices with other variables.csv")

# Assuming your data has columns: settlement_date, month, daytype, hour, price
# Assuming the target variable is 'price' and exogenous variables are 'month', 'daytype', 'hour'

# Convert settlement_date to datetime if it's not already in datetime format
data['SETTLEMENTDATE'] = pd.to_datetime(data['SETTLEMENTDATE'])

# Sort the data based on settlement_date if it's not already sorted
data = data.sort_values(by='SETTLEMENTDATE')

# Set settlement_date as the index
data.set_index('SETTLEMENTDATE', inplace=True)

# Create lag features for the target variable (price) if needed
data['price_lag1'] = data['RRP'].shift(1)
# Add more lag features as needed

# Drop rows with NaN values resulting from lag features
data = data.dropna()

# Define features and target variable
X = data[['MONTH', 'DAYTYPE', 'HOUR', 'price_lag1','YEAR','DAY']]  # Add other features as needed
y = data['RRP']

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train, test = data[0:train_size], data[train_size:]

X_train, X_test = X.iloc[:train_size, :], X.iloc[train_size:, :]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# Initialize and train the XGBoost model
model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)

# Streamlit app
st.title("XGBoost Model Deployment")

year = st.text_input('Enter Year')
month = st.text_input("Enter Month (1-12)")
day = st.text_input('Enter Day (1-31)')
hour = st.text_input("Enter Hour (0-23)")

# Submission button
if st.button("Get Prediction"):
    # Convert user input to appropriate data types
    year = int(year)
    month = int(month)
    day = int(day)
    hour = int(hour)

    # Create datetime object for the selected date
    selected_date = pd.to_datetime(f"{year}-{month:02d}-{day:02d} {hour:02d}:00:00")

    # Determine day type dynamically based on the selected date
    day_type = "Weekday" if selected_date.weekday() < 5 else "Weekend"
    day_type_encoded = 1 if day_type == 'Weekday' else 0

    # Calculate price_lag1 dynamically based on historical data up to the selected date
    historical_data = data.loc[data.index < selected_date]
    latest_price_lag1 = historical_data['RRP'].iloc[-1]

    # Convert user input into DataFrame for prediction
    input_data = pd.DataFrame({'MONTH': [month], 'DAYTYPE': [day_type_encoded], 'HOUR': [hour], 'price_lag1': [latest_price_lag1], 'YEAR': [year], 'DAY': [day]})

    # Make prediction on user input
    prediction = model.predict(input_data)

    # Display the prediction
    st.write(f"Predicted Price of the given input in NSW: {prediction[0]}")

    # Optional: Display model evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    st.write(f'Mean Squared Error of the model: {mse}')
    st.write(f'Mean Absolute Percentage Error of the model: {mape}')

