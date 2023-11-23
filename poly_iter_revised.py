import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

sp500 = '^GSPC'

tickers = ['AAPL', 'GOOG', 'T', 'F', 'AMZN']


start_date = '2021-01-01'
end_date = '2023-11-13'


df_500 = yf.download(sp500, start=start_date, end=end_date)['Adj Close']

df = yf.download(tickers, start=start_date, end=end_date)['Adj Close']


returns = df.pct_change().dropna()


risk_free_rate = 0.05


expected_returns = returns.mean()


cov_matrix = returns.cov()

# Equally weighted. Change as you please.
weights = [0.2, 0.2, 0.2, 0.2, 0.2]


portfolio_expected_return = np.dot(expected_returns, weights)


portfolio_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))


sharpe_ratio = (portfolio_expected_return - risk_free_rate) / portfolio_volatility


print('Portfolio Expected Return:', portfolio_expected_return)
print('Portfolio Volatility (Standard Deviation):', portfolio_volatility)
print('Sharpe Ratio:', sharpe_ratio)

# change degree as you please
degree = 2
poly = PolynomialFeatures(degree=degree,include_bias=False)
returns_poly = poly.fit_transform(returns)


correlation_matrix = pd.DataFrame(returns_poly).corr()


print('Correlation Matrix:')
print(correlation_matrix)


portfolio_value = (returns.dot(weights) + 1).cumprod()
plt.figure(figsize=(10,6))
plt.plot(portfolio_value.index, portfolio_value, label='Portfolio Value')
plt.title('Diversified Portfolio Performance')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.grid(True)
plt.legend()
plt.show()
