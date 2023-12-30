import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')
from pypfopt import expected_returns,risk_models,EfficientFrontier


sp500 = '^GSPC'

tickers = ['AAPL','GOOG','T','F','AMZN','TGT','SBUX']


start_date = '2021-01-01'
end_date = '2023-12-29'


df_500 = yf.download(sp500, start=start_date, end=end_date)['Close']

df = yf.download(tickers, start=start_date, end=end_date)['Close']


stock_returns = df.pct_change().dropna()


risk_free_rate = 0.05


mu_s = expected_returns.mean_historical_return(df)
cov_s = risk_models.sample_cov(df)

ef_stocks = EfficientFrontier(mu_s, cov_s)

# Equally weighted. Change as you please.
weights = ef_stocks.max_sharpe(risk_free_rate)
cleaned_weights = ef_stocks.clean_weights()



portfolio_expected_return = ef_stocks.portfolio_performance()[0]
print(portfolio_expected_return)

portfolio_volatility = ef_stocks.portfolio_performance()[1]


sharpe_ratio = (portfolio_expected_return - risk_free_rate) / portfolio_volatility

print('Portfolio Expected Return:', portfolio_expected_return)
print('Portfolio Volatility (Standard Deviation):', portfolio_volatility)
print('Sharpe Ratio:', sharpe_ratio)

# Change degree as you please, it does not change anything
degree = 3
poly = PolynomialFeatures(degree=degree, include_bias=False)
stock_returns_poly = poly.fit_transform(stock_returns)

correlation_matrix = pd.DataFrame(stock_returns_poly).corr()

print('Correlation Matrix:')
print(correlation_matrix)

# Performance
metrics_labels = ['Expected Return','Volatility','Sharpe Ratio']
metrics_values = [portfolio_expected_return,portfolio_volatility,sharpe_ratio]

plt.figure(figsize=(10,6))
plt.bar(metrics_labels,metrics_values,color=['blue','orange','green'])
plt.title('Diversified Portfolio Performance Metrics')
plt.ylabel('Value')
plt.show()
