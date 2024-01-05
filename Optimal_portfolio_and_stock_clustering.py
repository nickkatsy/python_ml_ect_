import yfinance as yf
import pandas as pd
from pypfopt import risk_models,EfficientFrontier,expected_returns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# importing a list of bond and stock tickers


stocks = ['AAPL','TGT','MCD','IBM','TSLA']
bonds = ['BND','HYG','TIP','IEF','LQD']


start_date = '2023-01-05'
end_date = '2024-01-05'

# import stocks and bonds from yfinance
stocks_df = yf.download(tickers=stocks,start=start_date,end=end_date)['Adj Close']
bonds_df = yf.download(tickers=bonds,start=start_date,end=end_date)['Adj Close']


#returns for stocks and bonds
stock_returns = stocks_df.pct_change().dropna()
bond_returns = bonds_df.pct_change().dropna()

# Expected Returns and Volatility for select Bonds,Stocks, and full Portfolio

# for stocks
mu_s = expected_returns.mean_historical_return(stocks_df)
S_cov = risk_models.sample_cov(stocks_df)

#for bonds

mu_bonds = expected_returns.mean_historical_return(bonds_df)
S_b = risk_models.sample_cov(bonds_df)


# Efficient Frontier for stocks bonds and complete portfolio

ef_stocks = EfficientFrontier(mu_s, S_cov)
ef_bonds = EfficientFrontier(mu_bonds, S_b)



#risk_free rate based on current ten-year treasury bonds

risk_free_rate = 0.04


#weights for stocks
weights_stocks = ef_stocks.max_sharpe()
cleaned_weights_stocks = ef_stocks.clean_weights()
print('weights for stock portfolio',cleaned_weights_stocks)


#weights for bonds


weights_bonds = ef_bonds.max_sharpe()
cleaned_weights_bonds = ef_bonds.clean_weights()
print('weights for bond portfolio: ',cleaned_weights_bonds)




#optimal portfolio for stocks

optimal_expected_return_stock_portfolio = ef_stocks.portfolio_performance()[0]
volatility_of_optimal_stock_portfolio = ef_stocks.portfolio_performance()[1]

sharpe_ratio_stocks = (optimal_expected_return_stock_portfolio - risk_free_rate) / volatility_of_optimal_stock_portfolio

print('Expected Returns on optimal risky(stock) portfolio: ', optimal_expected_return_stock_portfolio)
print('Standard Deviation (risk) of risky(stock) portfolio:',volatility_of_optimal_stock_portfolio)
print('Sharpe Ratio for optimal stock portfolio: ', sharpe_ratio_stocks)


# optimal portfolio for bonds
optimal_expected_returns_bond_portfolio = ef_bonds.portfolio_performance()[0]
volatility_of_optimal_bond_portfolio = ef_bonds.portfolio_performance()[1]

sharpe_ratio_bonds = (optimal_expected_returns_bond_portfolio - risk_free_rate) / volatility_of_optimal_bond_portfolio

print('Expected Returns on optimal risk-free(bond) portfolio: ', optimal_expected_returns_bond_portfolio)
print('Standard Deviation (risk) of risky(stock) portfolio:',volatility_of_optimal_bond_portfolio)
print('Sharpe Ratio for optimal stock portfolio: ', sharpe_ratio_bonds)



#correlation matrix for stocks and bonds
from sklearn.preprocessing import PolynomialFeatures

degree = 3
poly = PolynomialFeatures(degree=degree, include_bias=False)
stock_returns_poly = poly.fit_transform(stock_returns)

correlation_matrix_stocks = pd.DataFrame(stock_returns_poly).corr()

print('Correlation Matrix For Stocks:')
print(correlation_matrix_stocks)




bond_returns_poly = poly.fit_transform(bond_returns)
correlation_matrix_bonds = pd.DataFrame(bond_returns_poly).corr()
print('Correlation Matrix for Bonds',correlation_matrix_bonds)

# bar plot of portfolio performance for selected stocks and bonds



# visuals of optimal portfolio for selected stocks

labels = ['Expected Returns','Standard Deviation(Risk)','Sharpe Ratio']
values = [optimal_expected_return_stock_portfolio,volatility_of_optimal_stock_portfolio,sharpe_ratio_stocks]
plt.bar(labels,values,color=['blue','yellow','green'])
plt.title('Optimal Risky Portfolio (Stocks)')
plt.show()


# visuals of optimal portfolio for selected bonds


labels = ['Expected Returns','Standard Deviation(Risk)','Sharpe Ratio']
values = [optimal_expected_returns_bond_portfolio,volatility_of_optimal_bond_portfolio,sharpe_ratio_bonds]
plt.bar(labels,values,color=['blue','yellow','green'])
plt.title('Optimal Risk-free Portfolio (Bonds)')
plt.show()


#allocation for complete portfolio

#full portfolio
portfolio = pd.concat([stocks_df,bonds_df],axis=1,join='outer').dropna()
#returns for complete porfolio
portfolio_returns = portfolio.pct_change().dropna()

#expected_returns and standard deviaiton of complete portfolio

mu_p = expected_returns.mean_historical_return(portfolio)
S_p = risk_models.sample_cov(portfolio)

#efficient frontier complete portfolio

ef_portfolio = EfficientFrontier(mu_p, S_p)

#weights for complete portfolio
portfolio_weights = ef_portfolio.max_sharpe()
portfolio_weights_clean = ef_portfolio.clean_weights()

#optimal full Portfolio

# Discrete allocation to determine left over cash based on your allocation of portfolio value
from pypfopt.discrete_allocation import DiscreteAllocation,get_latest_prices

latest_prices = get_latest_prices(portfolio)
da = DiscreteAllocation(portfolio_weights_clean, latest_prices,total_portfolio_value=1000)

allocation, leftover = da.greedy_portfolio()
print('Discrete Allocation: ',allocation)
print('Left over cash',leftover)



# clusterting

df_stocks = yf.download(tickers=stocks,start=start_date,end=end_date)

features = df_stocks[['Adj Close','Close','High']]

features.describe()
features.isna().sum()


X = features


from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler()

X_scaled = ms.fit_transform(X)


from sklearn.cluster import KMeans

cc = []
for i in range(1,20):
    kmeans = KMeans(n_clusters=i,init='k-means++',n_init=10,random_state=42).fit(X_scaled)
    cc.append(kmeans.inertia_)
    

#Elbow Method to find where the clusters hit a maximium

plt.plot(range(1,20),cc,marker='*')
plt.xlabel('Clusters')
plt.ylabel('Inertia')
plt.title('The Elbow Method')
plt.show()


#optimal number of clusters is at k = 13

kmeans = KMeans(n_clusters=13,init='k-means++',n_init=10,random_state=42).fit(X_scaled)
labels = kmeans.fit_predict(X_scaled)
X['Cluster'] = labels


plt.scatter(X_scaled[:,0],X_scaled[:,1],c=X['Cluster'],s=300,marker='x',edgecolors='r')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],marker='o',color='m')
plt.show()



from sklearn.metrics import silhouette_score
sh = silhouette_score(X_scaled, labels)
print(f'The silhoutte score is {sh*100:.2f}%')
print(kmeans.inertia_)




