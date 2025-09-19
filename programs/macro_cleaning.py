import pandas as pd
import numpy as np

drilling = pd.read_csv('data/raw_data/macro/ppi_drilling_main.csv')
drilling['date'] = pd.to_datetime(drilling['date'])
support = pd.read_csv('data/raw_data/macro/ppi_support_activities.csv')
support['date'] = pd.to_datetime(support['date'])
oil = pd.read_csv('data/raw_data/macro/ppi_crude_oil.csv')
oil['date'] = pd.to_datetime(oil['date'])
gas = pd.read_csv('data/raw_data/macro/ppi_natural_gas.csv')
gas['date'] = pd.to_datetime(gas['date'])

macro = drilling.merge(support, on='date', how='left')
macro = macro.merge(oil, on='date', how='left')
macro = macro.merge(gas, on='date', how='left')

# Sort to ensure correct quarter-end selection
macro = macro.sort_values('date')

# Aggregate monthly series into calendar quarters (average and last-of-quarter)
macro = (
    macro.resample('Q', on='date')
         .mean()
)

#macro_q = pd.concat([macro_q_avg, macro_q_last], axis=1).reset_index()

macro['delta_log_oil'] = np.log(macro['ppi_oil']) - np.log(macro['ppi_oil'].shift(1))
macro['delta_log_gas'] = np.log(macro['ppi_gas']) - np.log(macro['ppi_gas'].shift(1))
macro['delta_log_drilling'] = np.log(macro['ppi_drilling']) - np.log(macro['ppi_drilling'].shift(1))
macro['delta_log_support'] = np.log(macro['ppi_support']) - np.log(macro['ppi_support'].shift(1))

# Add six lags for delta_log_oil only (for residual regressions)
for lag in [1, 2, 3, 4, 5, 6, 7, 8]:
    macro[f'delta_log_oil_lag{lag}'] = macro['delta_log_oil'].shift(lag)

# Add eight lags for the residuals
for lag in [1, 2, 3, 4, 5, 6]:
    macro[f'delta_log_drilling_{lag}'] = macro['delta_log_drilling'].shift(lag)
    macro[f'delta_log_support_{lag}'] = macro['delta_log_support'].shift(lag)


# Use quarter-end calendar dates in the date column and write output
macro.reset_index().to_csv('data/processed_data/macro.csv', index=False)
