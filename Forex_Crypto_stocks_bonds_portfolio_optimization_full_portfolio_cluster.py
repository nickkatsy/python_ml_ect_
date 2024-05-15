import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import yfinance as yf

stock_tickers = ['AAPL','GOOGL','F','T','TSLA','MSFT','AMZN','NVDA','NFLX','SBUX','TGT']
bond_tickers = ['BND','HYG','TIP','IEF','LQD','SHY','AGG','MUB','HYG','TLT']
forex_tickers = ['EURUSD=X', 'JPY=X', 'GBPUSD=X', 'AUDUSD=X', 'USDCAD=X', 'USDCHF=X', 'NZDUSD=X', 'EURGBP=X']
crypto_tickers = ['BTC-USD', 'ETH-USD', 'ADA-USD', 'XRP-USD', 'SOL-USD', 'DOT-USD', 'LTC-USD', 'LINK-USD', 'BCH-USD']

start_date = '2023-02-03'
end_date = '2024-05-14'

stocks_df = yf.download(stock_tickers,start=start_date,end=end_date)['Close']
forex = yf.download(forex_tickers,start=start_date,end=end_date)['Close']
cyp_tic = yf.download(tickers=crypto_tickers,start=start_date,end=end_date)['Close']
bond_df = yf.download(tickers=bond_tickers,start=start_date,end=end_date)['Close']

forex_returns = forex.pct_change().dropna()
cryto_returns = cyp_tic.pct_change().dropna()
stocks_returns = stocks_df.pct_change().dropna()
bond_returns = bond_df.pct_change().dropna()




from pypfopt import risk_models,EfficientFrontier,expected_returns

mu_stocks = expected_returns.mean_historical_return(stocks_df)
S_stocks = risk_models.sample_cov(stocks_df)
mu_bonds = expected_returns.mean_historical_return(bond_df)
S_bonds = risk_models.sample_cov(bond_df)
mu_forex = expected_returns.mean_historical_return(forex)
S_forex = risk_models.sample_cov(forex)
mu_crypto = expected_returns.mean_historical_return(cyp_tic)
S_crypto = risk_models.sample_cov(cyp_tic)


ef_stocks = EfficientFrontier(mu_stocks, S_stocks)
weights_stocks = ef_stocks.max_sharpe(risk_free_rate=.04)
cleaned_weights_stocks = ef_stocks.clean_weights()
print('stock weights',cleaned_weights_stocks)

expected_returns_stock_portfolio = ef_stocks.portfolio_performance()[0]
risk_portfolio_performance_stocks = ef_stocks.portfolio_performance()[1]
print('portfolio perfomance (Efficient Frontier of risky assets',expected_returns_stock_portfolio)
print('standard deviation(risk) of stocks',risk_portfolio_performance_stocks)




ef_bonds = EfficientFrontier(mu_bonds, S_bonds)
weights_bonds = ef_bonds.max_sharpe(0.001)
bond_portfolio_expected_returns = ef_bonds.portfolio_performance()[0]
bond_portfolio_risk = ef_bonds.portfolio_performance()[1]
print('weights of bond portfolio: ',weights_bonds)
print('returns on risk-free(ish not munis) portfolio',bond_portfolio_expected_returns)
print('volatility of bond portfolio',bond_portfolio_risk)



ef_forex = EfficientFrontier(mu_forex, S_forex)
weights_forex = ef_forex.max_sharpe(.05)
forex_portfolio_performance_expected_returns = ef_forex.portfolio_performance()[0]
forex_portfolio_risk = ef_forex.portfolio_performance()[1]
print('weights from the Forex',weights_forex)
print('Expected returns from Forex portfolio: ',forex_portfolio_performance_expected_returns)
print('forex risk: ',forex_portfolio_risk)



ef_crypto = EfficientFrontier(mu_crypto, S_crypto)
weights_crypto = ef_crypto.max_sharpe(risk_free_rate=0.0)

portfolio_performance_crypto_returns = ef_crypto.portfolio_performance()[0]
portfolio_performance_risk_crypto = ef_crypto.portfolio_performance()[1]
print('portfolio weights for dogecoin or whatever',weights_crypto)
print('expected returns crypto currency portfolio',portfolio_performance_crypto_returns)
print('crpyto risk',portfolio_performance_risk_crypto)


sharpe_stocks = (expected_returns_stock_portfolio - .04) / risk_portfolio_performance_stocks
sharpe_bonds = (bond_portfolio_expected_returns - .001) / bond_portfolio_risk
sharpe_forex = (forex_portfolio_performance_expected_returns - .05) / forex_portfolio_risk
sharpe_crypto = (portfolio_performance_crypto_returns - 0.0) / portfolio_performance_risk_crypto

print("Sharpe Ratio for Stocks:", sharpe_stocks)
print("Sharpe Ratio for Bonds:", sharpe_bonds)
print("Sharpe Ratio for Forex:", sharpe_forex)
print("Sharpe Ratio for Crypto:", sharpe_crypto)


print('expected returns for stock portfolio',expected_returns_stock_portfolio)
print('expectedd returns bond portfolio',bond_portfolio_expected_returns)
print('expected returns forex',forex_portfolio_performance_expected_returns)
print('expect crypto returns',portfolio_performance_crypto_returns)


import matplotlib.pyplot as plt
#visuals of expected returns for bonds,stocks, forex market and crypto currency
labels_er = ['stocks','bonds',"forex","crypto"]
values_er = [expected_returns_stock_portfolio,bond_portfolio_expected_returns,forex_portfolio_performance_expected_returns,portfolio_performance_crypto_returns]
plt.bar(labels_er,values_er)
plt.show()


labels_risk = ['stocks','bonds','forex','crypto']
values_risk = [risk_portfolio_performance_stocks,bond_portfolio_risk,forex_portfolio_risk,portfolio_performance_risk_crypto]
plt.bar(labels_risk,values_risk)
plt.show()




labels_sharpe = ["stocks","bonds","forex","crypto"]
values_sharpe = [sharpe_stocks,sharpe_bonds,sharpe_forex,sharpe_crypto]
plt.bar(labels_sharpe,values_sharpe)
plt.show()



#full portfolio performance
full_portfolio = pd.concat([stocks_df,bond_df,forex,cyp_tic], axis=1, join='outer')
portfolio_returns = full_portfolio.pct_change().dropna()
print('returns from full portfolio',portfolio_returns)

# expected returns full portfolio

mu_full_portfolio = expected_returns.mean_historical_return(full_portfolio)
S_full_portfolio = risk_models.sample_cov(full_portfolio)

#efficient frontier full portfolio

ef_full = EfficientFrontier(mu_full_portfolio,S_full_portfolio)
weights_full = ef_full.max_sharpe()
weights_full_cleaned = ef_full.clean_weights()
print('weights',weights_full_cleaned)

#for fun

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=3)

poly_returns = poly.fit_transform(portfolio_returns)
poly_returns_df = pd.DataFrame(poly_returns)
print(poly_returns_df)

full_portfolio_performance_expected_returns = ef_full.portfolio_performance()[0]
full_portfolio_risk = ef_full.portfolio_performance()[1]


sharpe_full_portfolio = full_portfolio_performance_expected_returns - .05 / full_portfolio_risk


print(full_portfolio_performance_expected_returns)
print(full_portfolio_risk)
print(sharpe_full_portfolio)

def full_portfolio_plot(full_portfolio_performance_expected_returns,full_portfolio_risk,sharpe_full_portfolio):
    labels = ['Expected Returns ','risk','sharpe ratio ']
    values = [full_portfolio_performance_expected_returns,full_portfolio_risk,sharpe_full_portfolio]
    plt.bar(labels,values)
    plt.title('Full Portfolio performance')
    plt.xlabel('Expected Returns Sharpe Ratio')
    plt.show()



full_portfolio_plot(full_portfolio_performance_expected_returns, full_portfolio_risk, sharpe_full_portfolio)



#clustering full porfolio


df_stocks = yf.download(stock_tickers,start=start_date,end=end_date)
forex_df = yf.download(forex_tickers,start=start_date,end=end_date)
cyp_df = yf.download(tickers=crypto_tickers,start=start_date,end=end_date)
bond = yf.download(tickers=bond_tickers,start=start_date,end=end_date)


full = pd.concat([df_stocks,forex_df,cyp_df,bond],axis=1,join='outer')
full.describe()

X_full = full[['High','Low','Adj Close','Volume']]

X_full.dropna(inplace=True)

from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler()
X_scaled = ms.fit_transform(X_full)

from sklearn.cluster import KMeans

cc = []
for i in range(1,21):
    kmeans = KMeans(n_clusters=i,init='k-means++',n_init=20,random_state=0).fit(X_scaled)
    cc.append(kmeans.inertia_)




#elbow method
plt.plot(range(1,21),cc,marker='X')
plt.show()

# k = 12

kmeans = KMeans(n_clusters=12,init='k-means++', n_init=20, random_state=1).fit(X_scaled)
labels = kmeans.labels_



X_full['Cluster 1'] = labels
X_full['Cluster 2'] = labels
X_full['Cluster 3'] = labels
X_full['Cluster 4'] = labels


plt.scatter(X_scaled[:,0],X_scaled[:,1],c=X_full['Cluster 1'],edgecolors='r')
plt.scatter(X_scaled[:,0],X_scaled[:,1],c=X_full['Cluster 2'],edgecolors='m')
plt.scatter(X_scaled[:,0],X_scaled[:,1],c=X_full['Cluster 3'],edgecolors='r')
plt.scatter(X_scaled[:,0],X_scaled[:,1],c=X_full['Cluster 4'],edgecolors='m')
plt.show()


#scoring

from sklearn.metrics import silhouette_score

sh = silhouette_score(X_scaled,labels)
print('sh score: ',sh*100)
print(kmeans.inertia_)



