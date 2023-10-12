import yfinance as yf
import pandas as pd
from sklearn.cluster import KMeans
from pypfopt import expected_returns, risk_models, EfficientFrontier

# tickers
tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']

# select any date you want
start_date = '2020-01-01'
end_date = '2023-10-11'

# Create an empty DataFrame to store historical stock price data
historical_data = pd.DataFrame()

# Retrieve historical data for each ticker
for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date)
    historical_data[ticker] = data['Adj Close']

# Perform K-Means clustering on the historical data
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
historical_data['Cluster'] = kmeans.fit_predict(historical_data)

# risk-free rate
risk_free_rate = 0.03  

# Function to optimize the portfolio for a given cluster
def optimize_portfolio_for_cluster(cluster_data, risk_free_rate):
    mu = expected_returns.mean_historical_return(cluster_data)
    S = risk_models.sample_cov(cluster_data)
    ef = EfficientFrontier(mu, S)
    ef.add_constraint(lambda w: w >= 0)
    ef.efficient_risk(risk_free_rate)
    cleaned_weights = ef.clean_weights()
    return cleaned_weights

# dictionary to store the optimized portfolios for each cluster
optimized_portfolios = {}

# Optimize portfolios for each cluster
for cluster_id in range(n_clusters):
    cluster_data = historical_data[historical_data['Cluster'] == cluster_id].drop('Cluster', axis=1)
    optimized_weights = optimize_portfolio_for_cluster(cluster_data, risk_free_rate)
    optimized_portfolios[f'Cluster {cluster_id}'] = optimized_weights

# Print the optimized portfolio weights for each cluster
for cluster, weights in optimized_portfolios.items():
    print(f'{cluster} Portfolio Weights:')
    print(weights)
    
    

import matplotlib.pyplot as plt

mu = expected_returns.mean_historical_return(cluster_data)
S = risk_models.sample_cov(cluster_data)
ef = EfficientFrontier(mu, S)
ef.efficient_frontier()
plt.figure(figsize=(10, 6))
ef.plot_efficient_frontier()
plt.title('Efficient Frontier for Cluster 0')
plt.xlabel('Expected Return')
plt.ylabel('Risk (Standard Deviation)')

plt.show()



