import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from pypfopt import expected_returns,risk_models,EfficientFrontier
import yfinance as yf

#link to speedrun: https://www.youtube.com/watch?v=2uOCKN-KcSo
stocks = ['IBM','MCD','AAPL','GOOGL','TSLA']

bonds = ['BND','HYG','TIP','IEF','LQD']


start_date = '2020-10-10'
end_date = '2024-01-04'


stocks_df = yf.download(tickers=stocks,start=start_date,end=end_date)['Close']
bonds_df = yf.download(tickers=bonds,start=start_date,end=end_date)['Close']


portfolio = pd.concat([stocks_df,bonds_df],axis=1,join='inner')

portfolio_returns = portfolio.pct_change().dropna()

mu_portfolio = expected_returns.mean_historical_return(portfolio)
S_portfolio = risk_models.sample_cov(portfolio)


risk_free_rate = 0.03

ef_portfolio = EfficientFrontier(mu_portfolio, S_portfolio)
weights = ef_portfolio.max_sharpe(risk_free_rate)
cleaned_weights = ef_portfolio.clean_weights()

optimal_porfolio_returned = ef_portfolio.portfolio_performance()[0]
volatility_portfolio = ef_portfolio.portfolio_performance()[1]


sharpe_ratio =  optimal_porfolio_returned - risk_free_rate  / volatility_portfolio

print('sharpe',sharpe_ratio)
print('expected returns full portfolio: ',optimal_porfolio_returned)
print('standard deviation full potrfolio: ',volatility_portfolio)

import matplotlib.pyplot as plt

labels = ['Expected Returns','Risk','Sharpe Ratio']
values = [optimal_porfolio_returned,volatility_portfolio,sharpe_ratio]
plt.bar(labels,values,color=['blue','yellow','blue'])
plt.show()


sp500 = '^GSPC'

market = yf.download(tickers=sp500,start=start_date,end=end_date)['Close']

import statsmodels.api as sm

model_tes = sm.OLS(exog=stocks_df['AAPL'],endog=market).fit()
print(model_tes.summary())

returns_stocks = stocks_df.pct_change().dropna()


market_returns = market.pct_change().dropna()

X = returns_stocks
y = market_returns

from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)


from sklearn.linear_model import LinearRegression

lr = LinearRegression()

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,BaggingRegressor
rfr = RandomForestRegressor()
gbr = GradientBoostingRegressor()
BR = BaggingRegressor()

from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor()


from sklearn.metrics import r2_score,mean_squared_error

def evaluate_stuffff(model,X_train,X_test,y_train,y_test):
    model = model.fit(X_train,y_train)
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    print(f'{model.__class__.__name__}, --R2-- {r2*100:.2f}%; --MSE-- {mse:.2f}')
    return pred


lr_pred = evaluate_stuffff(lr, X_train, X_test, y_train, y_test)
rfr_pred = evaluate_stuffff(rfr, X_train, X_test, y_train, y_test)
gbr_pred = evaluate_stuffff(gbr, X_train, X_test, y_train, y_test)
BR_pred = evaluate_stuffff(BR, X_train, X_test, y_train, y_test)
tree_pred = evaluate_stuffff(tree, X_train, X_test, y_train, y_test)







#clustering#####
df_stocks = yf.download(tickers=stocks,start=start_date,end=end_date)
features = df_stocks[['Close','High','Low']]


features.info()
features.isna().sum()

from sklearn.preprocessing import MinMaxScaler

ms = MinMaxScaler()

scaled = ms.fit_transform(features)


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_scaled = pca.fit_transform(scaled)


from sklearn.cluster import KMeans

cc = []
for i in range(2,12):
    kmeans = KMeans(n_clusters=i,init='k-means++',n_init=20,random_state=42).fit(pca_scaled)
    cc.append(kmeans.inertia_)


#elbow method

plt.plot(range(2,12),cc,marker='*')
plt.xlabel('Clusters')
plt.ylabel('inertia')
plt.show()




kmeans = KMeans(n_clusters=11,init='k-means++',n_init=20,random_state=42).fit(pca_scaled)
labels = kmeans.labels_
features['clusters'] = labels


plt.scatter(pca_scaled[:,0],pca_scaled[:,1],c=features['clusters'],marker='x',edgecolors='m')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],marker='*',color='c')
plt.show()



pred = kmeans.predict(pca_scaled)
from sklearn.metrics import silhouette_score
sh = silhouette_score(pca_scaled,labels)
print(f'the sh score: {sh*100:.2f}%')
print(kmeans.inertia_)







