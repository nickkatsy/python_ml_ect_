import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define a list of stock tickers
stock_tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']

# Set the date range for historical data
start_date = '2021-01-01'
end_date = '2023-10-10'

# Download historical data for the selected stocks
df = yf.download(stock_tickers, start=start_date, end=end_date)['Adj Close']

# Calculate daily returns
returns = df.pct_change().dropna()

# Define risk-free rate
risk_free_rate = 0.015  

# Calculate expected returns for each stock
expected_returns = returns.mean()

# Calculate the covariance matrix
cov_matrix = returns.cov()

# Portfolio weights (adjust however you want)
weights = [0.2, 0.2, 0.2, 0.2, 0.2]

# Portfolio expected return
portfolio_expected_return = np.dot(expected_returns, weights)

# Portfolio standard deviation (volatility)
portfolio_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))

# Calculate the Sharpe ratio
sharpe_ratio = (portfolio_expected_return - risk_free_rate) / portfolio_volatility

# Print the results
print("Portfolio Expected Return:", portfolio_expected_return)
print("Portfolio Volatility (Standard Deviation):", portfolio_volatility)
print("Sharpe Ratio:", sharpe_ratio)




portfolio_value = (returns.dot(weights) + 1).cumprod()
plt.figure(figsize=(10, 6))
plt.plot(portfolio_value.index, portfolio_value, label="Portfolio Value", color='b')
plt.title("Diversified Portfolio Performance")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.grid(True)
plt.legend()
plt.show()






