import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm

# 1. Generate Hypothetical Dataset
np.random.seed(42)
date_rng = pd.date_range(start='2021-01-01', end='2024-12-31', freq='MS')
n_months = len(date_rng)

# Trend, seasonality, and noise for car sales
trend = np.linspace(100, 150, n_months)
seasonality = 15 * np.sin(np.arange(n_months) * (2 * np.pi / 12) + np.pi / 2) # Summer peak
noise = np.random.normal(0, 5, n_months)
car_sales = pd.Series(trend + seasonality + noise, index=date_rng, name='Car_Sales')

# Exogenous variable: GDP
gdp = pd.Series(np.linspace(22, 25, n_months) + np.random.normal(0, 0.2, n_months), index=date_rng, name='GDP')

df = pd.concat([car_sales, gdp], axis=1)

# 2. Plot Original Data
print('--- Original Data ---')
df.plot(figsize=(12, 6), title='Hypothetical US Car Sales and GDP (2021-2024)')
plt.show()

# 3. Test for Stationarity
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

# Make the series stationary
df['Car_Sales_Diff'] = df['Car_Sales'].diff(12).dropna() # Seasonal differencing
adf_test(df['Car_Sales_Diff'], 'Seasonally Differenced Car Sales')

# 4. Choose p, q, P, Q order
fig, axes = plt.subplots(1, 2, figsize=(16, 4))
plot_acf(df['Car_Sales_Diff'].dropna(), lags=15, ax=axes[0], title='ACF')
plot_pacf(df['Car_Sales_Diff'].dropna(), lags=15, ax=axes[1], title='PACF')
plt.show()

# Based on the plots, we can make an initial guess for the orders.
# ACF: Cuts off after lag 1, so q=1.
# PACF: Cuts off after lag 1, so p=1.
# Seasonal lags (12) in ACF suggest Q=1.
# Seasonal lags (12) in PACF suggest P=1.
p, d, q = 1, 0, 1
P, D, Q, s = 1, 1, 1, 12

# 5. Fit the SARIMAX model
train = df.iloc[:-3] # Train on all data except the last 3 months of 2024
test = df.iloc[-3:]  # Test on the last 3 months of 2024

model = sm.tsa.SARIMAX(train['Car_Sales'], exog=train['GDP'], order=(p, d, q), seasonal_order=(P, D, Q, s))
results = model.fit()

print('--- SARIMAX Model Summary ---')
print(results.summary())

# 6. Forecast and Plot
# Forecast for the last 3 months of 2024
forecast_2024 = results.get_forecast(steps=3, exog=test['GDP'])
forecast_ci_2024 = forecast_2024.conf_int()

# Forecast for summer 2023
start_date_2023 = '2023-06-01'
end_date_2023 = '2023-08-01'
forecast_2023 = results.get_prediction(start=start_date_2023, end=end_date_2023, exog=df.loc[start_date_2023:end_date_2023, 'GDP'])
forecast_ci_2023 = forecast_2023.conf_int()

# Plot the forecasts
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(df['Car_Sales'], label='Observed')

# Plot 2024 forecast
forecast_2024.predicted_mean.plot(ax=ax, label='Forecast 2024', style='--', color='red')
ax.fill_between(forecast_ci_2024.index, forecast_ci_2024.iloc[:, 0], forecast_ci_2024.iloc[:, 1], color='r', alpha=0.2)

# Plot 2023 forecast
forecast_2023.predicted_mean.plot(ax=ax, label='Forecast 2023', style='--', color='green')
ax.fill_between(forecast_ci_2023.index, forecast_ci_2023.iloc[:, 0], forecast_ci_2023.iloc[:, 1], color='g', alpha=0.2)

plt.title('Car Sales Forecast vs. Observed')
plt.legend()
plt.show()
