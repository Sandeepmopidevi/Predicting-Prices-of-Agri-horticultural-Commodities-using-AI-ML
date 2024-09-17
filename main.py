import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import streamlit as st

# Updated file path
file_path = "datamain.csv"  
df = pd.read_csv(file_path)

# Set index to 'Commodities' and transpose to make years the index
df.set_index('Commodities', inplace=True)
df = df.T

# Set date index for time series starting from January 2014
df.index = pd.date_range(start='2014-01', periods=len(df), freq='M')

# Fill missing data
df = df.ffill()

# List of commodities
commodities = df.columns.tolist()

# Streamlit app
st.title("Commodity Price Forecasting")

# Select commodity
selected_commodity = st.selectbox("Choose a Commodity", commodities)

if st.button("Submit"):
    data = df[selected_commodity]

    # Fit SARIMAX model (adjust parameters as necessary)
    model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
    sarimax_model = model.fit(disp=False)

    # Forecast for the next 5 years
    forecast = sarimax_model.get_forecast(steps=12 * 5)  # Forecast 5 years (12 months per year)
    forecasted_values = forecast.predicted_mean

    # Set future dates for forecast
    forecast_years = pd.date_range(start='2025-01', periods=12 * 5, freq='M')
    forecast_df = pd.DataFrame({'Year': forecast_years, f'{selected_commodity}_Price_Forecast': forecasted_values})

    # Display forecast table
    st.write(f"### {selected_commodity} Price Forecast (2025-2029)")
    st.write(forecast_df)

    # Plot actual and forecasted prices
    plt.figure(figsize=(10, 6))
    plt.plot(data, label=f'Actual {selected_commodity} Prices')
    plt.plot(forecast_years, forecasted_values, label=f'Forecasted {selected_commodity} Prices', color='orange')
    plt.title(f'{selected_commodity} Price Forecast (2025-2029)')
    plt.xlabel('Year')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)

    # Calculate training RMSE
    train_rmse = np.sqrt(((data - sarimax_model.fittedvalues) ** 2).mean())
    st.write(f"Training RMSE: {train_rmse:.4f}")
