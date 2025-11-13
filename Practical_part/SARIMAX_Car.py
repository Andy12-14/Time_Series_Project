import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm

# --- Setup for Consistent Data Generation ---
np.random.seed(42)

# 1. Generate Hypothetical Dataset (Extended to 2026)
date_rng = pd.date_range(start='2021-01-01', end='2026-12-31', freq='MS')
n_months = len(date_rng)

# Trend, seasonality, and noise for car sales
trend = np.linspace(100, 180, n_months) 
seasonality = 15 * np.sin(np.arange(n_months) * (2 * np.pi / 12) + np.pi / 2)
noise = np.random.normal(0, 5, n_months)
car_sales = pd.Series(trend + seasonality + noise, index=date_rng, name='Car_Sales')

# Exogenous variable: GDP (Extended into the future)
gdp = pd.Series(np.linspace(22, 28, n_months) + np.random.normal(0, 0.2, n_months), index=date_rng, name='GDP')

df = pd.concat([car_sales, gdp], axis=1)

# 2. Test for Stationarity (ADF Test) - Output to Console
def adf_test(series, name=''):
    print(f'--- ADF Test for {name} ---')
    result = adfuller(series.dropna())
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] <= 0.05:
        print('Conclusion: Series is Stationary')
    else:
        print('Conclusion: Series is Non-Stationary')

adf_test(df['Car_Sales'], 'Original Car Sales')
df['Car_Sales_Diff'] = df['Car_Sales'].diff(12).dropna()
adf_test(df['Car_Sales_Diff'], 'Seasonally Differenced Car Sales')

# 3. Choose p, q, P, Q order (ACF/PACF Plot)
p, d, q = 1, 0, 1
P, D, Q, s = 1, 1, 1, 12

# 4. Define Train, Test (2024), and Future Forecast Sets
# Train: Data up to the end of 2023
train = df.loc[:'2023-12-01'] 
# Test (2024): Actual data for the 12 months of 2024
test_2024 = df.loc['2024-01-01':'2024-12-01'] 
# Future Forecast: Hypothetical data from 2025-01-01 onward
forecast_data_future = df.loc['2025-01-01':] 

# 5. Fit the SARIMAX model using only the training data (up to 2023)
model = sm.tsa.SARIMAX(
    train['Car_Sales'], 
    exog=train['GDP'], 
    order=(p, d, q), 
    seasonal_order=(P, D, Q, s),
    enforce_stationarity=False,
    enforce_invertibility=False 
)
results = model.fit(disp=False) 

print('--- SARIMAX Model Summary (Trained up to 2023) ---')
print(results.summary())

# 6. Generate Forecasts

# A. Forecast for the entire year 2024 (Out-of-sample test: 12 steps)
forecast_2024 = results.get_forecast(steps=12, exog=test_2024['GDP'])
forecast_ci_2024 = forecast_2024.conf_int()

# B. Long-Term Forecast (2025-2026: 24 steps, starting from the end of the training data)
FORECAST_STEPS_TOTAL = 12 + 24 # 2024 + 2025 + 2026
# Get prediction starting from 2024-01-01
extended_forecast = results.get_forecast(steps=FORECAST_STEPS_TOTAL, exog=df.loc['2024-01-01':, 'GDP'])

# Extract only the 2025-2026 part for the future plot
future_forecast_2025_2026 = extended_forecast.predicted_mean.loc['2025-01-01':]
future_forecast_ci_2025_2026 = extended_forecast.conf_int().loc['2025-01-01':]


# =======================================================
# PLOT 1: 2024 Test Forecast (Validation)
# =======================================================
fig, ax = plt.subplots(figsize=(14, 7))
plt.title('Plot 1: SARIMAX Model Validation - 2024 Test Forecast vs. Actual')
ax.set_xlabel('Date')
ax.set_ylabel('Car Sales')
ax.grid(True, linestyle='--')

# 1. Observed Data (Full 2021-2026 dataset for context)
ax.plot(df['Car_Sales'], label='Observed Actual Data (2021-2026)', color='gray', alpha=0.6, linewidth=1.5)

# Highlight Training Data
ax.plot(train['Car_Sales'], label='Training Data (2021-2023)', color='blue', linewidth=2)

# 2. 2024 Test Forecast
forecast_2024.predicted_mean.plot(ax=ax, label='Forecast 2024 (Test Period)', style='--', color='red', linewidth=2)
ax.fill_between(
    forecast_ci_2024.index, 
    forecast_ci_2024.iloc[:, 0], 
    forecast_ci_2024.iloc[:, 1], 
    color='r', 
    alpha=0.3,
    label='95% Confidence Interval'
)

# Demarcation Line
ax.axvline(x=train.index[-1], color='k', linestyle=':', linewidth=1, label='Forecast Start (Jan 2024)')

plt.legend(loc='upper left')
plt.show()

# =======================================================
# PLOT 2: Long-Term Future Forecast (2025-2026)
# =======================================================
fig, ax = plt.subplots(figsize=(14, 7))
plt.title('Plot 2: SARIMAX Model - Long-Term Future Forecast (2025-2026)')
ax.set_xlabel('Date')
ax.set_ylabel('Car Sales')
ax.grid(True, linestyle='--')

# 1. Observed Data (Full 2021-2026 dataset for context)
ax.plot(df['Car_Sales'], label='Observed Actual Data (2021-2026)', color='gray', alpha=0.6, linewidth=1.5)

# Highlight Observed Data before Future Forecast
ax.plot(df.loc[:'2024-12-01']['Car_Sales'], label='Observed Data (2021-2024)', color='blue', linewidth=2)

# 2. 2025-2026 Long-Term Forecast
future_forecast_2025_2026.plot(ax=ax, label='Future Forecast (2025-2026)', style='--', color='green', linewidth=2)
ax.fill_between(
    future_forecast_ci_2025_2026.index, 
    future_forecast_ci_2025_2026.iloc[:, 0], 
    future_forecast_ci_2025_2026.iloc[:, 1], 
    color='g', 
    alpha=0.3,
    label='95% Confidence Interval'
)

# Demarcation Line
ax.axvline(x=test_2024.index[-1], color='k', linestyle=':', linewidth=1, label='Future Forecast Start (Jan 2025)')

plt.legend(loc='upper left')
plt.show()