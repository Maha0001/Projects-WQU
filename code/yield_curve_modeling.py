"""
Yield Curve Modeling with Nelson-Siegel and Cubic-Spline Models

This script models the yield curve using U.S. Treasury securities data. It includes:
- Data collection using the FRED API
- Fitting the Nelson-Siegel and Cubic-Spline models
- Plotting and comparing the models
- Saving the results as images
"""

import numpy as np
import pandas as pd
from fredapi import Fred
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import CubicSpline
from nelson_siegel_svensson.calibrate import calibrate_ns_ols
import os

sns.set()

# Initialize the FRED API with your key
fred = Fred(api_key="95eb212842318d85c6198945d6514bf4")

# List of Treasury yield series IDs
series_ids = ['DGS1MO', 'DGS3MO', 'DGS6MO', 'DGS1', 'DGS2', 'DGS3', 'DGS5', 'DGS7', 'DGS10', 'DGS20', 'DGS30']

# Function to get data for a single series
def get_yield_data(series_id):
    """Fetches yield data for a given series ID from the FRED API."""
    data = fred.get_series(series_id)
    return data

# Get the latest yield data
latest_yields = {series_id: get_yield_data(series_id).iloc[-1] for series_id in series_ids}

# Define maturity and yield variables
maturities = np.array([1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30])
yields = np.array([latest_yields[series_id] for series_id in series_ids])

# Fit the Nelson-Siegel model
ns_params = calibrate_ns_ols(maturities, yields)
nelson_siegel_curve, _ = ns_params  # Unpack parameters and metadata

# Access Nelson-Siegel parameters
beta0 = nelson_siegel_curve.beta0
beta1 = nelson_siegel_curve.beta1
beta2 = nelson_siegel_curve.beta2
tau = nelson_siegel_curve.tau

# Calculate Nelson-Siegel yields
ns_yields = (
    beta0
    + beta1 * (1 - np.exp(-maturities / tau)) / (maturities / tau)
    + beta2 * ((1 - np.exp(-maturities / tau)) / (maturities / tau) - np.exp(-maturities / tau))
)

# Plot the Nelson-Siegel results
plt.figure(figsize=(10, 6))
plt.plot(maturities, yields, 'o', label='Data')  # Original data points
plt.plot(maturities, ns_yields, label='Nelson-Siegel')  # Nelson-Siegel curve
plt.xlabel('Maturity (years)')
plt.ylabel('Yield')
plt.title('Nelson-Siegel Model Fit')
plt.legend()
os.makedirs("figure", exist_ok=True)
plt.savefig("figure/nelson_siegel_curve.png")
plt.show()

# Fit the Cubic-Spline model
cs = CubicSpline(maturities, yields)

# Plot the Cubic-Spline results
plt.figure(figsize=(10, 6))
plt.plot(maturities, yields, 'o', label='Data')
plt.plot(maturities, cs(maturities), label='Cubic Spline')
plt.xlabel('Maturity (years)')
plt.ylabel('Yield')
plt.title('Cubic Spline Model Fit')
plt.legend()
plt.savefig("figure/cubic_spline_plot.png")
plt.show()

"""
Ethical Considerations:
Smoothing data with these models is not inherently unethical if done transparently for legitimate purposes. However, misuse to manipulate or
mislead stakeholders is unethical. Transparency and honesty in data analysis are crucial.
"""
