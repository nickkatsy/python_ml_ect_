import pandas as pd
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('https://raw.githubusercontent.com/LeeMorinUCF/QMB6912S23/main/demo_10/Tractor_SampleSelection/Data/TRACTOR7.csv')

df.info()
df.isna().sum()
df.describe()
print(df.dtypes)


import seaborn as sns
import matplotlib.pyplot as plt



sns.heatmap(df.corr(), annot=True)
plt.show()



def subss(df):
    _,axs = plt.subplots(2,2,figsize=(10,6))
    sns.barplot(x='summer',y='saleprice',ax=axs[0,0],data=df)
    sns.lineplot(x='fwd',y='saleprice',ax=axs[0,1],data=df)
    sns.boxplot(x='winter',y='saleprice',ax=axs[1,0],data=df)
    sns.scatterplot(x='enghours',y='saleprice',ax=axs[1,1],data=df)
    plt.show()
    
subss(df)




X = df.drop('saleprice',axis=1)
y = df['saleprice']

import statsmodels.api as sm

model = sm.OLS(exog=sm.add_constant(X),endog=y).fit()
print(model.summary())

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

from sklearn.linear_model import LinearRegression,Ridge,Lasso
lr = LinearRegression()
ridge = Ridge()
lasso = Lasso()

from sklearn.ensemble import RandomForestRegressor,BaggingRegressor

rfr = RandomForestRegressor()
BR = BaggingRegressor()

from sklearn.metrics import r2_score,mean_squared_error

def evaluate_(X_train_scaled,X_test_scaled,y_train,y_test,model):
    model = model.fit(X_train_scaled,y_train)
    pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    print(f'{model.__class__.__name__}, --R2-- {r2*100:.2f}%; --MSE-- {mse:.2f}')
    return pred

lr_pred = evaluate_(X_train_scaled, X_test_scaled, y_train, y_test, lr)
ridge_pred = evaluate_(X_train_scaled, X_test_scaled, y_train, y_test,ridge)
lasso_pred = evaluate_(X_train_scaled, X_test_scaled, y_train, y_test, lasso)
rfr_pred = evaluate_(X_train_scaled,X_test_scaled,y_train,y_test,rfr)
BR_pred = evaluate_(X_train_scaled, X_test_scaled, y_train, y_test, BR)

from scipy.optimize import minimize
import numpy as np

def SSR_Linear_Regression(beta,X,y):
    model = LinearRegression()
    model.coef_ = beta[1:]  
    model.intercept_ = beta[0]
    pred = model.predict(X)
    residuals = y - pred
    ssr = np.sum(residuals**2)
    return ssr



initial_guess = np.zeros(X.shape[1] + 1)


result = minimize(SSR_Linear_Regression,x0=initial_guess,args=(X.values,y.values),method='Nelder-Mead')
optimal_params = result.x


optimal_lr_intercept = optimal_params[0]
optimal_lr_coefficients = optimal_params[1:]


optimal_lr_ = LinearRegression()
optimal_lr_.intercept_ = optimal_lr_intercept
optimal_lr_.coef_ = optimal_lr_coefficients

y_pred_lr_opt = optimal_lr_.predict(X)
mse_lr_opt = mean_squared_error(y, y_pred_lr_opt)
r2_lr_opt = r2_score(y, y_pred_lr_opt)



print(f'Optimal Coefficients model: {optimal_lr_.coef_}')
print(f'MSE for Optimized Model: {mse_lr_opt:.2}')
print(f'R^2 for Optimized Model: {r2_lr_opt*100:.2f}%')


# Regularization for ridge and lasso

def SSR_ridge(beta,X,y,lambda_ridge):
    pred = np.dot(X,beta)
    residuals = y - pred
    ssr = np.sum(residuals**2) + lambda_ridge * np.sum(beta**2)
    return ssr


def SSR_lasso(beta,X,y,lambda_lasso):
    pred = np.dot(X, beta)
    residuals = y - pred
    ssr = np.sum(residuals**2) + lambda_lasso * np.sum(np.abs(beta))
    return ssr




def optimize_ridge(lambda_ridge):
    result = minimize(SSR_ridge,x0=np.zeros(X.shape[1]),args=(X,y,lambda_ridge),method='Nelder-Mead')
    optimal_params = result.x
    return optimal_params

# Lasso Regression Optimization 
def optimize_lasso(lambda_lasso):
    result = minimize(SSR_lasso,x0=np.zeros(X.shape[1]),args=(X,y,lambda_lasso),method='Nelder-Mead')
    optimal_params = result.x
    return optimal_params

#  Ridge Optimization  Model
optimal_ridge_params = optimize_ridge(lambda_ridge=0.1)
optimal_ridge = Ridge(alpha=0.1)
optimal_ridge.coef_ = optimal_ridge_params[1:]
optimal_ridge.intercept_ = optimal_ridge_params[0]
optimal_ridge.fit(X, y)

#Lasso Optimization model
optimal_lasso_params = optimize_lasso(lambda_lasso=0.1)
optimal_lasso = Lasso(alpha=0.1)
optimal_lasso.coef_ = optimal_lasso_params[1:]
optimal_lasso.intercept_ = optimal_lasso_params[0]
optimal_lasso.fit(X,y)

#results for Ridge
print(f'Optimal Ridge Coefficients: {optimal_ridge.coef_}')
print(f'Ridge Intercept: {optimal_ridge.intercept_}')
y_pred_ridge = optimal_ridge.predict(X)
mse_ridge_opt = mean_squared_error(y,y_pred_ridge)
r2_ridge_opt = r2_score(y,y_pred_ridge)
print(f'MSE for Optimized Ridge Model: {mse_ridge_opt:.2f}')
print(f'R^2 for Optimized Ridge Model: {r2_ridge_opt*100:.2f}%')

# results for Lasso
print(f'Optimal Lasso Coefficients: {optimal_lasso.coef_}')
print(f'Lasso Intercept: {optimal_lasso.intercept_}')
y_pred_lasso_opt = optimal_lasso.predict(X)
mse_lasso_opt_util = mean_squared_error(y,y_pred_lasso_opt)
r2_lasso_opt_util = r2_score(y, y_pred_lasso_opt)
print(f'MSE for Optimized Lasso Model: {mse_lasso_opt_util:.2f}')
print(f'R^2 for Optimized Lasso Model: {r2_lasso_opt_util*100:.2f}%')



