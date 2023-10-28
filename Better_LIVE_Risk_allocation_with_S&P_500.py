import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from pypfopt import expected_returns, EfficientFrontier
import warnings
warnings.filterwarnings('ignore')

ticker = ['AAPL','GOOG','T','MGM','IBM','^GSPC','TSLA']


start_date = '2020-10-10'
end_date = '2023-10-26'

df = yf.download(tickers=ticker,start=start_date,end=end_date)['Adj Close']






def live_stock_tickers(df):
    plt_stuff,axs = plt.subplots(2,2,figsize=(10,6))
    sns.scatterplot(df,x='IBM',y='T',ax=axs[0,0])
    sns.scatterplot(df,x='AAPL',y='MGM',ax=axs[0,1])
    sns.boxplot(df['^GSPC'],ax=axs[1,0])
    sns.scatterplot(df,x='GOOG',y='^GSPC',ax=axs[1,1])
    plt.show()


live_stock_tickers(df)




# tickers and S&P 500
def SP_500(df):
    plt_2,axs2= plt.subplots(2,3,figsize=(10,6))
    sns.scatterplot(df,x='AAPL',y='^GSPC',ax=axs2[0,0])
    sns.scatterplot(df,x='MGM',y='^GSPC',ax=axs2[0,1])
    sns.scatterplot(df,x='TSLA',y='^GSPC',ax=axs2[0,2])
    sns.scatterplot(df,x='IBM',y='^GSPC',ax=axs2[1,0])
    sns.scatterplot(df,x='T',y='^GSPC',ax=axs2[1,1])
    sns.scatterplot(df,x='GOOG',y='^GSPC',ax=axs2[1,2])
    plt.show()



SP_500(df)

import statsmodels.api as sm


model_AAPL = sm.OLS(exog=df[['AAPL']],endog=df[['^GSPC']]).fit()
print(model_AAPL.summary())

model_MGM = sm.OLS(exog=df[['MGM']],endog=df[['^GSPC']]).fit()
print(model_MGM.summary())


model_TSLA = sm.OLS(exog=df[['TSLA']],endog=df[['^GSPC']]).fit()
print(model_TSLA.summary())



model_GOOG = sm.OLS(exog=df['GOOG'],endog=df[['^GSPC']]).fit()
print(model_GOOG.summary())


### All ticker training data together now


X = df.drop(['^GSPC'],axis=1)
y = df[['^GSPC']]

model_full = sm.OLS(exog=(X),endog=(y)).fit()
print(model_full.summary())


pre_pred = model_full.predict(X)

plt.scatter(y,pre_pred)


#train/test, models yada yada through 


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)


from sklearn.linear_model import LinearRegression

lr = LinearRegression().fit(X_train,y_train)
lr_pred = lr.predict(X_test)

from sklearn.svm import SVR

svr = SVR(degree=2).fit(X_train,y_train)
svr_pred = svr.predict(X_test)

from sklearn.ensemble import RandomForestRegressor

rfc = RandomForestRegressor().fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)

from sklearn.metrics import r2_score,mean_squared_error

# Linear Regression Results

mse_lr = mean_squared_error(y_test,lr_pred)
print(mse_lr)
r2_lr = r2_score(y_test, lr_pred)
print(r2_lr)

# Standard Vector Machine Results

mse_svm = mean_squared_error(y_test, svr_pred)
print(mse_svm)
r2_svm = r2_score(y_test, svr_pred)
print(r2_svm)

#Random Forest Results


mse_rfc = mean_squared_error(y_test, rfc_pred)
print(mse_rfc)

r_2_rfc = r2_score(y_test, rfc_pred)
print(r_2_rfc)

def scatter_plot_of_results(df):
    plt.scatter(lr_pred,y_test)
    plt.scatter(rfc_pred, y_test)
    svr_scatter = plt.scatter(svr_pred,y_test)
    plt.plot()


scatter_plot_of_results(df)

#portfolio stuff


import riskfolio as rp

port1 = df.copy()

port = rp.Portfolio(returns=port1)

method_mu = 'hist'
method_cov = 'hist'

port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)

model = 'Classic'
rm = 'MV'
obj = 'Sharpe'
hist = True
rf = 0.04
l = -5

w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)

ax = rp.plot_pie(w=w, title='Sharpe Mean Variance', others=0.05, nrow=25, cmap="tab20", height=6, width=10, ax=None)

points = 50

frontier = port.efficient_frontier(model=model, rm=rm, points=points, rf=rf, hist=hist)

label = 'Max Risk Adjusted Return Portfolio'
mu = port.mu
cov = port.cov
returns = port.returns

ax = rp.plot_frontier(w_frontier=frontier, mu=mu, cov=cov, returns=returns, rm=rm, rf=rf, alpha=0.05, cmap='viridis', w=w, label=label, marker='*', s=16, c='r', height=6, width=10, ax=None)

rm = 'CVaR'

w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)

print(w)

ax = rp.plot_pie(w=w, title='Sharpe Mean CVaR', others=0.05, nrow=25, cmap="tab20", height=6, width=10, ax=None)

points = 50

frontier = port.efficient_frontier(model=model, rm=rm, points=points, rf=rf, hist=hist)

label = 'Max Risk Adjusted Return Portfolio'

ax = rp.plot_frontier(w_frontier=frontier, mu=mu, cov=cov, returns=returns, rm=rm, rf=rf, alpha=0.05, cmap='viridis', w=w, label=label, marker='*', s=16, c='r', height=6, width=10, ax=None)

points = 50

frontier = port.efficient_frontier(model=model, rm=rm, points=points, rf=rf, hist=hist)

label = 'Max Risk Adjusted Return Portfolio'

ax = rp.plot_frontier(w_frontier=frontier, mu=mu, cov=cov, returns=returns, rm=rm, rf=rf, alpha=0.05, cmap='viridis', w=w, label=label, marker='*', s=16, c='r', height=6, width=10, ax=None)

model = 'Classic'
rm = 'MV'
obj = 'Sharpe'
rf = 0.05
l = 5

w.mean()
w.std()

ax = rp.plot_pie(w=w, title='Sharpe Mean Variance', others=0.05, nrow=25, cmap='tab20', height=6, width=10, ax=None)

points = 50

frontier = port.efficient_frontier(model=model, rm=rm, points=points, rf=rf, hist=hist)

label = 'Max Risk Adjusted Return Portfolio'
mu = port.mu
cov = port.cov
returns = port.returns

ax = rp.plot_frontier(w_frontier=frontier, mu=mu, cov=cov, returns=returns, rm=rm, rf=rf, alpha=0.05, cmap='viridis', w=w, label=label, marker='*', s=16, c='r', height=6, width=10, ax=None)

plt.plot(frontier.std())

