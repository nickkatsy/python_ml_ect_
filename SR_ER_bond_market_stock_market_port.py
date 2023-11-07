import yfinance as yf
import warnings
warnings.filterwarnings('ignore')
import numpy as np

stock_tickers = ['AAPL','GOOGL','F','T','TSLA','MSFT','AMZN','NVDA','NFLX']
bond_tickers = ['BND','HYG','TIP','IEF','LQD','SHY','AGG','MUB','HYG','TLT']

np.random.shuffle(stock_tickers)
np.random.shuffle(bond_tickers)

start_date = '2020-11-03'
end_date = '2023-11-06'

bond_df = yf.download(tickers=bond_tickers, start=start_date, end=end_date)['Adj Close']
stock_df = yf.download(tickers=stock_tickers, start=start_date, end=end_date)['Adj Close']

import matplotlib.pyplot as plt
import seaborn as sns
import riskfolio as rp

def subplots_stocks(stock_df):
    plt_,axs = plt.subplots(2,2,figsize=(10,6))
    sns.scatterplot(stock_df, x='AAPL', y='F', ax=axs[0,0])
    sns.scatterplot(stock_df, x='GOOGL', y='T', ax=axs[0,1])
    sns.scatterplot(stock_df, x='TSLA', y='AAPL', ax=axs[1,0])
    sns.scatterplot(stock_df, x='F', y='TSLA', ax=axs[1,1])
    plt.show()

subplots_stocks(stock_df)

def subplots_bonds(bond_df):
    plt__,axs2 = plt.subplots(2,2,figsize=(10,6))
    sns.scatterplot(bond_df, x='HYG', y='TIP', ax=axs2[0,0])
    sns.scatterplot(bond_df, x='IEF', y='LQD', ax=axs2[0,1])
    sns.scatterplot(bond_df, x='BND', y='IEF', ax=axs2[1,0])
    sns.scatterplot(bond_df, x='IEF', y='LQD', ax=axs2[1,1])
    plt.show()

subplots_bonds(bond_df)

import statsmodels.api as sm

model_appl_BND = sm.OLS(exog=stock_df[['AAPL']], endog=bond_df[['BND']]).fit()
print(model_appl_BND.summary())

model_GOOG_HYG = sm.OLS(exog=stock_df[['GOOGL']], endog=bond_df[['HYG']]).fit()
print(model_GOOG_HYG.summary())

model_F_TIP = sm.OLS(exog=stock_df[['F']], endog=bond_df[['TIP']]).fit()
print(model_F_TIP.summary())

model_T_IEF = sm.OLS(exog=stock_df[['T']], endog=bond_df[['IEF']]).fit()
print(model_T_IEF.summary())

model_tsla_lqd = sm.OLS(exog=stock_df[['TSLA']], endog=bond_df[['LQD']]).fit()
print(model_tsla_lqd.summary())

X = stock_df
y = bond_df

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X_train, y_train)
lr_pred = lr.predict(X_test)

from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, lr_pred)
mse = mean_squared_error(y_test, lr_pred)
print('R-squared:', r2)
print('Mean Squared Error:', mse)

plt.plot(y_test, lr_pred)

returns_stocks = stock_df.pct_change().dropna()
returns_bonds = bond_df.pct_change().dropna()

assets = stock_df.copy()
debt = bond_df.copy()
port = rp.Portfolio(returns=assets)

method_mu = 'hist'
method_cov = 'hist'
port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)

model = 'Classic'
rm = 'MV'
obj = 'Sharpe'
hist = True
rf = 0.05
l = 5

w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
points = 50
frontier = port.efficient_frontier(model=model, rm=rm, points=points, rf=rf, hist=hist)
label = 'Max Risk-Adjusted Return Portfolio'

ax = rp.plot_frontier(w_frontier=frontier, mu=port.mu, cov=port.cov, returns=port.returns, rm=rm, rf=rf, alpha=0.05, cmap='viridis', w=w, label=label, marker='*', s=16, c='r', height=6, width=10, ax=None)
ax.legend()

plt.show()

cov_matrix_stocks_bonds = np.cov(returns_stocks.T, returns_bonds.T)
print(f'Covariance Matrix between Stocks and Bonds:\n{cov_matrix_stocks_bonds}')

import itertools


risk_free_rate = 0.05

combinations = list(itertools.product(stock_tickers, bond_tickers))

best_expected_return = -float('inf')
best_sharpe_ratio = -float('inf')
best_expected_return_combinations = []
best_sharpe_ratio_combinations = []

returns = {}

# Calculate returns for all combinations and find the best ones
for combination in combinations:
    stock, bond = combination
    subset_returns_stocks = returns_stocks[stock]
    subset_returns_bonds = returns_bonds[bond]
    combined_returns = subset_returns_stocks - subset_returns_bonds

    excess_returns = combined_returns - risk_free_rate
    sharpe_ratio = excess_returns.mean() / excess_returns.std()

    returns[combination] = combined_returns.mean()

    if sharpe_ratio > best_sharpe_ratio:
        best_sharpe_ratio = sharpe_ratio
        best_sharpe_ratio_combinations = [combination]
    elif sharpe_ratio == best_sharpe_ratio:
        best_sharpe_ratio_combinations.append(combination)

    if returns[combination] > best_expected_return:
        best_expected_return = returns[combination]
        best_expected_return_combinations = [combination]
    elif returns[combination] == best_expected_return:
        best_expected_return_combinations.append(combination)

# Determine which is higher and print the corresponding combinations
if best_sharpe_ratio >= best_expected_return:
    print('Combinations that maximize Sharpe ratio:')
    for combination in best_sharpe_ratio_combinations:
        print('Combination:', combination)
    print('Max Sharpe Ratio:', best_sharpe_ratio)

    print('Combinations that maximize expected returns:')
    for combination in best_expected_return_combinations:
        print('Combination:', combination)
    print('Max Expected Returns:', best_expected_return)
else:
    print('Combinations that maximize expected returns:')
    for combination in best_expected_return_combinations:
        print('Combination:', combination)
    print('Max Expected Returns:', best_expected_return)

    print('Combinations that maximize Sharpe ratio:')
    for combination in best_sharpe_ratio_combinations:
        print('Combination:', combination)
    print('Max Sharpe Ratio:', best_sharpe_ratio)