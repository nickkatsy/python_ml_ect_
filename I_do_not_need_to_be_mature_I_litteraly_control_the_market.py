import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from pypfopt import expected_returns,risk_models,EfficientFrontier


stocks = ['AAPL','TGT','SBUX','TGT','MCD','GOOGL','MCD']
sp50000 = '^GSPC'

start_date = '2016-10-12'
end_date = '2023-12-29'


df_stocks = yf.download(tickers=stocks,start=start_date,end=end_date)['Adj Close']
df_sp500 = yf.download(tickers=sp50000,start=start_date,end=end_date)['Adj Close']
returns_df = df_stocks.pct_change().dropna()
returns_sp500 = df_sp500.pct_change().dropna()

mu_s = expected_returns.mean_historical_return(df_stocks)
cov_S = risk_models.sample_cov(df_stocks)

ef_stocks = EfficientFrontier(mu_s, cov_S)

weights = ef_stocks.max_sharpe(risk_free_rate=0.05)
cleaned_weights = ef_stocks.clean_weights()

portfolio_expected_returns = ef_stocks.portfolio_performance()[0]
portfolio_volatility = ef_stocks.portfolio_performance()[1]


sharpe_ratio = portfolio_expected_returns - 0.05 / portfolio_volatility
print('expected returns:',portfolio_expected_returns)
print('portfolio standard deviation(risk)------',portfolio_volatility)
print('William Sharpe is pogged out and pissed off',sharpe_ratio)


from sklearn.preprocessing import PolynomialFeatures

poly_returns = PolynomialFeatures(degree=2,include_bias=False)

poly_r = poly_returns.fit_transform(returns_df)
correlationsss = pd.DataFrame(poly_r).corr()
print(correlationsss)

labels = ['Expected_returns','Volatility','Sharpe Ratio']
values = [portfolio_expected_returns,portfolio_volatility,sharpe_ratio]

plt.bar(labels,values,color=['blue','yellow','green'])
plt.show()

X = returns_df
y = returns_sp500

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
rfr = RandomForestRegressor()
gbr = GradientBoostingRegressor()


from sklearn.linear_model import Lasso,Ridge
lasso = Lasso()
ridge = Ridge()

from sklearn.metrics import r2_score,mean_squared_error

def evallll(model,X_train_scaled,X_test_scaled,y_train,y_test):
    model = model.fit(X_train_scaled,y_train)
    pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    print(f'{model.__class__.__name__}, --r2-- {r2*100:.2f}%; --MSE-- {mse*0.75:}%')
    return pred

lr_pred = evallll(lr, X_train_scaled, X_test_scaled, y_train, y_test)
rfr_pred = evallll(rfr, X_train_scaled, X_test_scaled, y_train, y_test)
gbr_pred = evallll(gbr, X_train_scaled, X_test_scaled, y_train, y_test)
lass_pred = evallll(lasso, X_train_scaled, X_test_scaled, y_train, y_test)
ridge_pred = evallll(ridge, X_train_scaled, X_test_scaled, y_train, y_test)






