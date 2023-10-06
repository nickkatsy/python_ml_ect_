import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import riskfolio as rp

df = pd.read_csv('https://raw.githubusercontent.com/nickkatsy/python_ml_ect_/master/capm.csv')
df.info()

df.describe()
df.nunique()


import matplotlib.pyplot as plt
import seaborn as sns

df.corr()


sns.heatmap(df.corr(), annot=True)


plt_stuff,axs = plt.subplots(2,2,figsize=(10,6))
plt_ = sns.scatterplot(df,x='BA',y='T',ax=axs[0,0])
plt_2 = sns.scatterplot(df,x='AAPL',y='BA',ax=axs[0,1])
plt_3 = sns.boxplot(df['sp500'],ax=axs[1,0])
plt_4 = sns.scatterplot(df,x='GOOG',y='sp500',ax=axs[1,1])

df1 = df.drop('Date',axis=1)


df1.info()



# tickers and S&P 500

plt_2,axss= plt.subplots(2,3,figsize=(10,6))
AAPL__ = sns.scatterplot(df1,x='AAPL',y='sp500',ax=axss[0,0])
MGM__ = sns.scatterplot(df1,x='MGM',y='sp500',ax=axss[0,1])
TSLA__ = sns.scatterplot(df1,x='TSLA',y='sp500',ax=axss[1,0])
IBM___ = sns.scatterplot(df1,x='IBM',y='sp500',ax=axss[1,1])
AMZN__ = sns.scatterplot(df1,x='AMZN',y='sp500',ax=axss[1,2])
GOOG_ = sns.scatterplot(df1,x='GOOG',y='sp500',ax=axss[0,2])


import statsmodels.api as sm


model_AAPL = sm.OLS(exog=df1[['AAPL']],endog=df1[['sp500']]).fit()
print(model_AAPL.summary())

model_MGM = sm.OLS(exog=df1[['MGM']],endog=df1[['sp500']]).fit()
print(model_MGM.summary())


model_TSLA = sm.OLS(exog=df1[['TSLA']],endog=df1[['sp500']]).fit()
print(model_TSLA.summary())



model_GOOG = sm.OLS(exog=df1['GOOG'],endog=df1[['sp500']]).fit()
print(model_GOOG.summary())


### All ticker training data together now


X = df1.drop(['T','BA','sp500'],axis=1)
y = df1[['sp500']]

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

svr = SVR().fit(X_train,y_train)
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


lr_scatter = plt.scatter(lr_pred,y_test)


rfc_scatter = plt.scatter(rfc_pred, y_test)




svr_scatter = plt.scatter(svr_pred,y_test)




#portfolio stuff

y_port = df1.drop(['T','BA'],axis=1)




port = rp.Portfolio(returns=y_port)

method_mu='hist' 
method_cov='hist'


port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)


model='Classic'
rm = 'MV'
obj = 'Sharpe'
hist = True
rf = 0
l = 0

w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)


ax = rp.plot_pie(w=w, title='Sharpe Mean Variance', others=0.05, nrow=25, cmap = "tab20",
                 height=6, width=10, ax=None)



points = 50

frontier = port.efficient_frontier(model=model, rm=rm, points=points, rf=rf, hist=hist)



label = 'Max Risk Adjusted Return Portfolio'
mu = port.mu # Expected returns
cov = port.cov # Covariance matrix
returns = port.returns # Returns of the assets

ax = rp.plot_frontier(w_frontier=frontier, mu=mu, cov=cov, returns=returns, rm=rm,
                      rf=rf, alpha=0.05, cmap='viridis', w=w, label=label,
                      marker='*', s=16, c='r', height=6, width=10, ax=None)


# Estimating risk portfolios

rm = 'CVaR' # Risk measure

w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)

print(w)


ax = rp.plot_pie(w=w, title='Sharpe Mean CVaR', others=0.05, nrow=25, cmap = "tab20",
                 height=6, width=10, ax=None)


points = 50 # Number of points of the frontier

frontier = port.efficient_frontier(model=model, rm=rm, points=points, rf=rf, hist=hist)

label = 'Max Risk Adjusted Return Portfolio'

ax = rp.plot_frontier(w_frontier=frontier, mu=mu, cov=cov, returns=returns, rm=rm,
                      rf=rf, alpha=0.05, cmap='viridis', w=w, label=label,
                      marker='*', s=16, c='r', height=6, width=10, ax=None)


points = 50 # Number of points of the frontier

frontier = port.efficient_frontier(model=model, rm=rm, points=points, rf=rf, hist=hist)


label = 'Max Risk Adjusted Return Portfolio'

ax = rp.plot_frontier(w_frontier=frontier, mu=mu, cov=cov, returns=returns, rm=rm,
                      rf=rf, alpha=0.05, cmap='viridis', w=w, label=label,
                      marker='*', s=16, c='r', height=6, width=10, ax=None)



model = 'Classic'
rm = 'MV'
obj = 'Sharpe'
rf = 0

w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
ax = rp.plot_pie(w=w, title='Sharpe Mean Variance', others=0.05, nrow=25, cmap = "tab20",
                 height=6, width=10, ax=None)




#constraints

assets = y_port[['GOOG','AAPL','AMZN','MGM','IBM','TSLA']]


y_assets = assets.pct_change().dropna()

port = rp.Portfolio(returns=y_assets)

# Calculating optimal portfolio

# parameters

method_mu='hist' # Method to estimate expected returns 
method_cov='hist' # Method to estimate covariance

port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)

# Estimate optimal portfolio:

model='Classic'
rm = 'MV' # Risk measure used
obj = 'Sharpe' # Objective function
hist = True # historical data,maybe
rf = 0 # Risk free rate
l = 0 # Risk aversion factor, only useful when obj is 'Utility'

w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)


w.mean()
w.std()



ax = rp.plot_pie(w=w, title='Sharpe Mean Variance', others=0.05, nrow=25, cmap = "tab20",
                 height=6, width=10, ax=None)


points = 50 # Number of points of the frontier

frontier = port.efficient_frontier(model=model, rm=rm, points=points, rf=rf, hist=hist)


label = 'Max Risk Adjusted Return Portfolio'
mu = port.mu # Expected returns
cov = port.cov # Covariance matrix
returns = port.returns # Returns of the assets

ax = rp.plot_frontier(w_frontier=frontier, mu=mu, cov=cov, returns=returns, rm=rm,
                      rf=rf, alpha=0.05, cmap='viridis', w=w, label=label,
                      marker='*', s=16, c='r', height=6, width=10, ax=None)




ax = rp.plot_frontier(w_frontier=frontier, mu=mu, cov=cov, returns=returns, rm='MDD',
                      rf=rf, alpha=0.05, cmap='viridis', w=w, label=label,
                      marker='*', s=16, c='r', height=6, width=10, ax=None)








