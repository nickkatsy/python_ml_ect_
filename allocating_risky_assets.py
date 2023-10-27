import yfinance as yf
from pypfopt import risk_models, EfficientFrontier, expected_returns
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Define the tickers for stocks and treasury bonds
tickers = ['^GSPC', '^IXIC', '^DJI', '^FTSE', '^GDAXI']
treasury_bonds = ['^TNX', '^TYX', 'IEF', 'AGG', 'BND']

# date range
start_date = '2013-10-21'
end_date = '2023-10-10'

# historical data for equities and bonds
equity_df = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
bond_df = yf.download(treasury_bonds, start=start_date, end=end_date)['Adj Close']

# daily returns for bonds and equities
returns_equity = equity_df.pct_change().dropna()
returns_bonds = bond_df.pct_change().dropna()

# the expected returns and covariance matrix for equities
mu_equity = expected_returns.mean_historical_return(returns_equity)
cov_matrix_equity = risk_models.sample_cov(returns_equity)

# mean returns for bonds
mu_bonds = expected_returns.mean_historical_return(returns_bonds)

# risk-free rate for return of bonds
risk_free_rate = mu_bonds.mean()

# EfficientFrontier for the equities
ef = EfficientFrontier(mu_equity, cov_matrix_equity)

# minimum volatility portfolio
weights_equity = ef.min_volatility()

# Portfolio expected return and volatility
portfolio_expected_return = ef.portfolio_performance()[0]
portfolio_volatility = ef.portfolio_performance()[1]

# Results
print('Optimal Weights (Equities):', weights_equity)
print('Portfolio Expected Return:', portfolio_expected_return)
print('Portfolio Volatility (Standard Deviation):', portfolio_volatility)

# Correlation matrix of equities
correlation_matrix = returns_equity.corr()



# correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matrix of Equities')
plt.show()

# Correlation matrix of bonds
correlation_matrix_debt = returns_bonds.corr()

# Plot the correlation matrix of bonds using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_debt, annot=True)
plt.title('Correlation Matrix of Bonds')
plt.show()
