"""
Exploiting Correlation in Financial Data

This script performs the following tasks:
1. Generate synthetic data and perform PCA.
2. Collect real financial data and perform PCA.
3. Compare the results between synthetic and real-world data.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from fredapi import Fred

# Synthetic Data Analysis

# Generate 5 uncorrelated Gaussian random variables
def generate_synthetic_data(n_samples=100, n_variables=5, mean=0, std_dev=0.01, seed=0):
    """Generates synthetic Gaussian random data."""
    np.random.seed(seed)
    return np.random.normal(mean, std_dev, (n_samples, n_variables))

data = generate_synthetic_data()

# Standardize the data
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data)

# Perform PCA on synthetic data
pca_synthetic = PCA()
pca_synthetic.fit(data_standardized)
explained_variance_synthetic = pca_synthetic.explained_variance_ratio_

# Scree plot for synthetic data
plt.figure(figsize=(10, 7))
plt.plot(np.cumsum(explained_variance_synthetic), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Scree Plot for Synthetic Data')
os.makedirs("figure", exist_ok=True)
plt.savefig("figure/scree_plot_synthetic_data.png")
plt.show()

# Real Data Analysis

# Collect daily closing yields for 5 government securities using FRED API
fred = Fred(api_key="95eb212842318d85c6198945d6514bf4")
series_ids = ['DGS1MO', 'DGS3MO', 'DGS6MO', 'DGS1', 'DGS2']

end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
start_date = (pd.Timestamp.today() - pd.DateOffset(months=6)).strftime('%Y-%m-%d')

def collect_yield_data(series_ids, start_date, end_date):
    """Fetches daily yields for given FRED series IDs."""
    data = {}
    for series_id in series_ids:
        data[series_id] = fred.get_series(series_id, start_date, end_date)
    return pd.DataFrame(data)

df = collect_yield_data(series_ids, start_date, end_date)
df.to_csv('data/us_government_securities_yields.csv')

# Calculate daily changes
daily_changes = df.diff()

# Perform PCA on the correlation matrix of real data
corr_matrix = daily_changes.corr()
pca_real = PCA(n_components=len(series_ids))
pca_real.fit(corr_matrix)
explained_variance_real = pca_real.explained_variance_ratio_

# Scree plot for real data
plt.figure(figsize=(10, 7))
plt.plot(np.cumsum(explained_variance_real), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Scree Plot for Real Data')
plt.savefig("figure/scree_plot_real_data.png")
plt.show()

"""
Comparison:
- Synthetic data shows a more uniform distribution of variance across components.
- Real-world data exhibits structured variance, with the first few components explaining most variance.

References:
1. Federal Reserve Economic Data (FRED): https://fred.stlouisfed.org/
2. PCA Documentation: https://scikit-learn.org/
"""
