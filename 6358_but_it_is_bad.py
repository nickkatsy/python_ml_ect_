import pandas as pd
import warnings
warnings.filterwarnings('ignore')

applications = pd.read_csv('https://raw.githubusercontent.com/LeeMorinUCF/QMB6358F23/main/final_exam_2021/applications.csv')


applications.describe()
applications.isna().sum()
applications.drop_duplicates(inplace=True)

df_app = applications.copy()

#drop ssn,app_id, and zip_code

df_app = df_app.drop(['ssn','zip_code','app_id'],axis=1)

df_app['homeownership'] = pd.get_dummies(df_app.homeownership,prefix='homeownership').iloc[:,0:1]

X = df_app.drop('purchases',axis=1)
y = df_app['purchases']


import statsmodels.api as sm
model1 = sm.OLS(exog=X,endog=y).fit()
print(model1.summary())



# train/test split model evaluation

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)



from sklearn.preprocessing import MinMaxScaler

ms = MinMaxScaler()

X_train_scaled = ms.fit_transform(X_train)
X_test_scaled = ms.transform(X_test)


from sklearn.linear_model import LinearRegression, Lasso,Ridge
lr = LinearRegression()
ridge = Ridge(alpha=.1)
lasso = Lasso(alpha=2)

# import enseble methods to see the difference betweeb linear methods
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor

rfr = RandomForestRegressor()
gbr = GradientBoostingRegressor()
BR = BaggingRegressor()




from sklearn.metrics import r2_score,mean_squared_error

def evaluate_first_model(model,X_train_scaled,y_train,X_test_scaled,y_test):
    model = model.fit(X_train_scaled,y_train)
    pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, pred)
    mse = mean_squared_error(y_test,pred)
    print(f'{model.__class__.__name__}, --R2-- {r2*100:.2f}%; --MSE-- {mse/80:.2f}')
    return pred

lr_pred = evaluate_first_model(lr, X_train_scaled, y_train, X_test_scaled, y_test)
rfr_pred = evaluate_first_model(rfr, X_train_scaled, y_train, X_test_scaled, y_test)
gbr_pred = evaluate_first_model(gbr,X_train_scaled,y_train,X_test_scaled,y_test)
BR_pred = evaluate_first_model(BR, X_train_scaled, y_train, X_test_scaled, y_test)
ridge_pred = evaluate_first_model(ridge, X_train_scaled, y_train, X_test_scaled, y_test)
lasso_pred = evaluate_first_model(lasso, X_train_scaled, y_train, X_test_scaled, y_test)

#import credit dataset

credit = pd.read_csv('https://raw.githubusercontent.com/LeeMorinUCF/QMB6358F23/main/final_exam_2021/credit_bureau.csv')

credit.isna().sum()
print(credit.dtypes)
credit.drop_duplicates(inplace=True)
credit.describe()

df_credit = credit.copy()

# create new dataset
purch_app = pd.concat([df_app,df_credit],axis=1,join='inner')

purch_app.info()
purch_app.isna().sum()
purch_app.drop_duplicates(inplace=True)

df_purch = purch_app.copy()

# drop app_id,ssn and zip_code

df_purch = df_purch.drop(['zip_code','ssn'],axis=1)

df_purch.isna().sum()

X_purch = df_purch.drop('purchases',axis=1)
y_purch = df_purch['purchases']

model2 = sm.OLS(exog=X,endog=y).fit()
print(model2.summary())


X_purch_train,X_purch_test,y_purch_test,y_purch_train = train_test_split(X_purch,y_purch,test_size=.20,random_state=42)




lr_purch = LinearRegression()
ridge_purch = Ridge(.1)
lasso_purch = Lasso(2)



rfr_purch = RandomForestRegressor()
gbr_purch = GradientBoostingRegressor()
BR_purch = BaggingRegressor()







X_train_purch,X_test_purch,y_train_purch,y_test_purch = train_test_split(X,y,test_size=.20,random_state=42)

from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler()


X_train_purch_scaled = ms.fit_transform(X_train_purch)
X_test_purch_scaled = ms.transform(X_test_purch)


def evaluate_purch(model_purch,X_train_purch_scaled,X_test_purch_scaled,y_train_purch,y_test_purch):
    model_purch = model_purch.fit(X_train,y_train)
    pred = model_purch.predict(X_test_scaled)
    r2_purch = r2_score(y_test, pred)
    mse_purch = mean_squared_error(y_test, pred)
    print(f'{model_purch.__class__.__name__}, --R2-- {r2_purch*100:.2f}, --MSE-- {mse_purch/80:.2f}%')
    return pred

lr_purch_pred = evaluate_purch(lr_purch, X_train_purch_scaled, X_test_purch_scaled, y_train_purch, y_test_purch)
lasso_purch_pred = evaluate_purch(lasso_purch, X_train_purch_scaled, X_test_purch_scaled, y_train_purch, y_test_purch)
ridge_purch_pred = evaluate_purch(ridge_purch, X_train_purch_scaled, X_test_purch_scaled, y_train_purch, y_test_purch)
rfr_purch_pred = evaluate_purch(rfr_purch, X_train_purch_scaled, X_test_purch_scaled, y_train_purch, y_test_purch)
gbr_purch_pred = evaluate_purch(gbr_purch, X_train_purch_scaled, X_test_purch_scaled, y_train_purch, y_test_purch)
BR_purch_pred = evaluate_purch(BR_purch, X_train_purch_scaled, X_test_purch_scaled, y_train_purch, y_test_purch)


#import demographic dataset
demographic = pd.read_csv('https://raw.githubusercontent.com/LeeMorinUCF/QMB6358F23/main/final_exam_2021/demographic.csv')
print(demographic.info())
demographic.isna().sum()
df_dem = demographic.copy()

df_dem.isna().sum()
df_dem.drop_duplicates(inplace=True)

#merging all three datasets

purch_full = pd.concat([df_dem,df_app,df_credit],axis=1,join='inner')
purch_full.isna().sum()
print(purch_full.dtypes)
purch_full.drop_duplicates(inplace=True)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,6))
sns.heatmap(purch_full.corr(),annot=True)
plt.show()

#drop ssn,zip_code,app_id


purch_full = purch_full.drop(['ssn','zip_code'],axis=1)

purch_full.info()
purch_full.drop_duplicates(inplace=True)
purch_full.isna().sum()



#visuals for all three datasets combined


def desc_full(purch_full):
    _,axs = plt.subplots(2,2,figsize=(10,6))
    sns.scatterplot(x='credit_limit',y='income',ax=axs[0,0],data=purch_full)
    axs[0,0].set_title('Credit Limit VS Average Income')
    
    
    sns.violinplot(x='num_late',y='homeownership',ax=axs[0,1],data=purch_full)
    axs[0,1].set_title('Numer of Late payments vs Homeownership')
    
    
    sns.barplot(x='num_bankruptcy',y='credit_limit',ax=axs[1,0],data=purch_full)
    axs[1,0].set_title('Credit limit vs Number of Individuals who filed for BankRuptcy')
    
    
    sns.barplot(x='num_late',y='income',ax=axs[1,1],data=purch_full)
    
    plt.show()


desc_full(purch_full)








X_full = purch_full.drop('purchases',axis=1)
y_full = purch_full['purchases']

model_full = sm.OLS(exog=X_full,endog=y_full).fit()
print(model_full.summary())


X_full_train,X_full_test,y_full_train,y_full_test = train_test_split(X_full,y_full,test_size=.20,random_state=42)





lr_full = LinearRegression()
ridge_full = Ridge(.1)
lasso_full = Lasso(2)



rfr_full = RandomForestRegressor()
gbr_full = GradientBoostingRegressor()
BR_full = BaggingRegressor()





def full_model(model_full,X_full_train,X_full_test,y_full_train,y_full_test):
    model_full = model_full.fit(X_full_train,y_full_train)
    pred = model_full.predict(X_full_test)
    r2_full = r2_score(y_full_test, pred)
    mse_full = mean_squared_error(y_full_test, pred)
    print(f'{model_full.__class__.__name__}, --r2-- {r2_full*100:.2f}%; --MSE-- {mse_full/80:.2f}')
    return pred


lr_full_pred = full_model(lr_full, X_full_train, X_full_test, y_full_train, y_full_test)
ridge_full_pred = full_model(ridge_full, X_full_train, X_full_test, y_full_train, y_full_test)
lasso_full_pred = full_model(lasso_full, X_full_train, X_full_test, y_full_train, y_full_test)
rfr_full_pred = full_model(rfr_full, X_full_train, X_full_test, y_full_train, y_full_test)
gbr_full_pred = full_model(gbr_full, X_full_train, X_full_test, y_full_train, y_full_test) 
BR_full_pred = full_model(BR_full, X_full_train, X_full_test, y_full_train, y_full_test)





#no train/test split
lr_full = LinearRegression()
ridge_full = Ridge()
lasso_full = Lasso()



rfr_full = RandomForestRegressor()
gbr_full = GradientBoostingRegressor()
BR_full = BaggingRegressor()







def full_model(model_full,X_full,y_full):
    model_full = model_full.fit(X_full,y_full)
    pred = model_full.predict(X_full)
    r2_full = r2_score(y_full, pred)
    mse_full = mean_squared_error(y_full, pred)
    print(f'{model_full.__class__.__name__}, --r2-- {r2_full*100:.2f}%; --MSE-- {mse_full/80:.2f}')
    return pred


lr_full_pred = full_model(lr_full, X_full, y_full)
ridge_full_pred = full_model(ridge_full, X_full, y_full)
lasso_full_pred = full_model(lasso_full, X_full, y_full)
rfr_full_pred = full_model(rfr, X_full, y_full)
gbr_full_pred = full_model(gbr, X_full, y_full)
BR_full_pred = full_model(BR_full, X_full, y_full)


#create utilization variable

utilization = purch_full['purchases'] / purch_full['credit_limit']

utilization.info()
utilization.isna().sum()
utilization.describe()


X_util = purch_full
y_util = utilization

model_util = sm.OLS(exog=X_util,endog=y_util).fit()
print(model_util.summary())



lr_util = LinearRegression()
ridge_util = Ridge(alpha=.1)
lasso_util = Lasso(alpha=2)


rfr_util = RandomForestRegressor()
gbr_util = GradientBoostingRegressor()
BR_util = BaggingRegressor()

def evaluate_util(model_util,X_util,y_util):
    model_util = model_util.fit(X_util,y_util)
    pred = model_util.predict(X_util)
    r2_util = r2_score(y_util, pred)
    mse_util = mean_squared_error(y_util,pred)
    print(f'{model_util.__class__.__name__}, --R2== {r2_util*100:.2f}%; --MSE-- {mse_util:.2f}')
    return pred


lr_util = evaluate_util(lr_util, X_util, y_util)
ridge_util_pred = evaluate_util(ridge_util, X_util, y_util)
lasso_util_pred = evaluate_util(lasso_util, X_util, y_util)
rfr_util_pred = evaluate_util(rfr, X_util, y_util)
gbr_util_pred = evaluate_util(gbr_util, X_util, y_util)
BR_util_pred = evaluate_util(BR_util, X_util, y_util)



import numpy as np

log_odds_util = np.log(utilization) / 1 - utilization

y_log_util = log_odds_util


lr_log_util = LinearRegression()

ridge_log_util = Ridge(alpha=.1)
lasso_log_util = Lasso(alpha=2)
rfr_log_util = RandomForestRegressor()
gbr_log_util = GradientBoostingRegressor()
BR_log_util = BaggingRegressor()




def evaluate_log_util(model_util_log,X_util,y_log_util):
    model_util_log = model_util_log.fit(X_util,y_log_util)
    pred = model_util_log.predict(X_util)
    r2_util_log = r2_score(y_log_util, pred)
    mse_util_log = mean_squared_error(y_log_util,pred)
    print(f'{model_util_log.__class__.__name__}, --R2== {r2_util_log*100:.2f}%; --MSE-- {mse_util_log:.2f}')
    return pred


lr_log_util_pred = evaluate_log_util(lr_log_util, X_util, y_log_util)
ridge_log_util_pred = evaluate_log_util(ridge_log_util, X_util, y_log_util)
lasso_log_util_pred = evaluate_log_util(lasso_log_util, X_util, y_log_util)
rfr_log_util_pred = evaluate_log_util(rfr_log_util, X_util, y_log_util)
gbr_log_util_pred = evaluate_log_util(gbr_log_util, X_util, y_log_util)
BR_log_util_pred = evaluate_log_util(BR_log_util, X_util, y_log_util)


# Not doing this for any models that are not linear
# no ensemble methoda, pretty much just linear regression, Lasso, and Ridge
#optimization part scipy SSR stuff


from scipy.optimize import minimize


def SSR(beta,X,y):
    pred = np.dot(beta,X)
    residuals = y - pred
    ssr = np.sum(residuals) ** 2
    return ssr




def SSR_Linear_Regression(beta,X,y):
    model = LinearRegression()
    model.coef_ = beta[1:]  
    model.intercept_ = beta[0]
    pred = model.predict(X)
    residuals = y - pred
    ssr = np.sum(residuals**2)
    return ssr



initial_guess = np.zeros(X_util.shape[1] + 1)


result = minimize(SSR_Linear_Regression,x0=initial_guess,args=(X_util.values,y_util.values),method='Nelder-Mead')
optimal_params = result.x


optimal_lr_util_intercept = optimal_params[0]
optimal_lr_util_coefficients = optimal_params[1:]


optimal_lr_util = LinearRegression()
optimal_lr_util.intercept_ = optimal_lr_util_intercept
optimal_lr_util.coef_ = optimal_lr_util_coefficients

y_pred_lr_util_opt = optimal_lr_util.predict(X_util)
mse_lr_util_opt = mean_squared_error(y_util, y_pred_lr_util_opt)
r2_lr_util_opt = r2_score(y_util, y_pred_lr_util_opt)

# util' results (linear regression)

print(f'Optimal Coefficients Utiliy model: {optimal_lr_util.coef_}')
print(f'MSE for Optimized Utiliy Model: {mse_lr_util_opt:.2}')
print(f'R^2 for Optimized Utility Model: {r2_lr_util_opt*100:.2f}%')


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


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_util_scaled = sc.fit_transform(X_util)


def optimize_ridge_utiliy(lambda_ridge):
    result = minimize(SSR_ridge,x0=np.zeros(X_util_scaled.shape[1]),args=(X_util_scaled,y_util,lambda_ridge),method='Nelder-Mead')
    optimal_params = result.x
    return optimal_params

# Lasso Regression Optimization utiliy model
def optimize_lasso_utility(lambda_lasso):
    result = minimize(SSR_lasso,x0=np.zeros(X_util_scaled.shape[1]),args=(X_util_scaled,y_util,lambda_lasso),method='Nelder-Mead')
    optimal_params = result.x
    return optimal_params

# Perform Ridge Optimization Utility Model
optimal_ridge_util_params = optimize_ridge_utiliy(lambda_ridge=0.1)
optimal_ridge_util = Ridge(alpha=0.1)
optimal_ridge_util.coef_ = optimal_ridge_util_params[1:]
optimal_ridge_util.intercept_ = optimal_ridge_util_params[0]
optimal_ridge_util.fit(X_util_scaled, y_util)

#Lasso Optimization utility model
optimal_lasso_util_params = optimize_lasso_utility(lambda_lasso=0.1)
optimal_lasso_util = Lasso(alpha=0.1)
optimal_lasso_util.coef_ = optimal_lasso_util_params[1:]
optimal_lasso_util.intercept_ = optimal_lasso_util_params[0]
optimal_lasso_util.fit(X_util_scaled,y_util)

# Print results for Ridge
print(f'Optimal Ridge Coefficients: {optimal_ridge_util.coef_}')
print(f'Ridge Intercept: {optimal_ridge_util.intercept_}')
y_pred_ridge_opt_util = optimal_ridge_util.predict(X_util_scaled)
mse_ridge_opt_util = mean_squared_error(y_util,y_pred_ridge_opt_util)
r2_ridge_opt = r2_score(y_util,y_pred_ridge_opt_util)
print(f'MSE for Optimized Ridge Model: {mse_ridge_opt_util:.2f}')
print(f'R^2 for Optimized Ridge Model: {r2_ridge_opt*100:.2f}%')

# results for Lasso
print(f'Optimal Lasso Coefficients: {optimal_lasso_util.coef_}')
print(f'Lasso Intercept: {optimal_lasso_util.intercept_}')
y_pred_lasso_opt_util = optimal_lasso_util.predict(X_util_scaled)
mse_lasso_opt_util = mean_squared_error(y_util,y_pred_lasso_opt_util)
r2_lasso_opt_util = r2_score(y_util, y_pred_lasso_opt_util)
print(f'MSE for Optimized Lasso Model: {mse_lasso_opt_util:.2f}')
print(f'R^2 for Optimized Lasso Model: {r2_lasso_opt_util*100:.2f}%')



#linear regression, ridge and lasso for log_util model


initial_guess = np.zeros(X_util.shape[1] + 1)

result = minimize(SSR_Linear_Regression,x0=initial_guess,args=(X_util.values,y_log_util.values),method='Nelder-Mead')
optimal_log_util_params = result.x

optimal_lr_util_log_intercept = optimal_log_util_params[0]
optimal_lr_util_log_coefficients = optimal_log_util_params[1:]

optimal_lr_util_log = lr_log_util
optimal_lr_util_log.intercept_ = optimal_lr_util_log_intercept
optimal_lr_util_log.coef_ = optimal_lr_util_log_coefficients

y_pred_lr_util_log_opt = optimal_lr_util_log.predict(X_util)
mse_lr_util_log_opt = mean_squared_error(y_log_util,y_pred_lr_util_log_opt)
r2_lr_util_log_opt = r2_score(y_log_util,y_pred_lr_util_log_opt)

# log util' results (linear regression)

print(f'Optimal Coefficients Utility model: {optimal_lr_util_log.coef_}')
print(f'MSE for Optimized Utility Model: {mse_lr_util_log_opt:.2f}')
print(f'R^2 for Optimized Utility Log Model: {r2_lr_util_log_opt*100:.2f}%')



# ridge and lasso log

def optimize_ridge_log_utiliy(lambda_ridge):
    result = minimize(SSR_ridge,x0=np.zeros(X_util_scaled.shape[1]),args=(X_util_scaled,y_log_util,lambda_ridge),method='Nelder-Mead')
    optimal_params = result.x
    return optimal_params

# Lasso Regression Optimization log utiliy model
def optimize_lasso_log_utility(lambda_lasso):
    result = minimize(SSR_lasso,x0=np.zeros(X_util_scaled.shape[1]),args=(X_util_scaled,y_log_util,lambda_lasso),method='Nelder-Mead')
    optimal_params = result.x
    return optimal_params



optimal_ridge_log_util_params = optimize_ridge_log_utiliy(lambda_ridge=0.1)
optimal_ridge_log_util = Ridge(alpha=0.1)
optimal_ridge_log_util.coef_ = optimal_ridge_log_util_params[1:]
optimal_ridge_log_util.intercept_ = optimal_ridge_log_util_params[0]
optimal_ridge_log_util.fit(X_util_scaled, y_log_util)

# Lasso Optimization log utility model
optimal_lasso_util_log_params = optimize_lasso_log_utility(lambda_lasso=0.1)
optimal_lasso_util_log = Lasso(alpha=0.1)
optimal_lasso_util_log.coef_ = optimal_lasso_util_log_params[1:]
optimal_lasso_util_log.intercept_ = optimal_lasso_util_log_params[0]
optimal_lasso_util_log.fit(X_util_scaled, y_log_util)

# Results for Ridge log
print(f'Optimal Ridge log Coefficients: {optimal_ridge_log_util.coef_}')
print(f'Ridge log Intercept: {optimal_ridge_log_util.intercept_}')
y_pred_ridge_log_opt_util = optimal_ridge_log_util.predict(X_util_scaled)
mse_ridge_log_opt_util = mean_squared_error(y_log_util, y_pred_ridge_log_opt_util)
r2_ridge_log_opt = r2_score(y_log_util, y_pred_ridge_log_opt_util)
print(f'MSE for Optimized log Ridge utility Model: {mse_ridge_log_opt_util:.2f}')
print(f'R^2 for Optimized log Ridge utility Model: {r2_ridge_log_opt * 100:.2f}%')

# Results for Lasso log
print(f'Optimal Lasso log Coefficients: {optimal_lasso_util_log.coef_}')
print(f'Lasso Intercept: {optimal_lasso_util_log.intercept_}')
y_pred_lasso_opt_util_log = optimal_lasso_util_log.predict(X_util_scaled)
mse_lasso_opt_util_log = mean_squared_error(y_log_util, y_pred_lasso_opt_util_log)
r2_lasso_opt_util_log = r2_score(y_log_util, y_pred_lasso_opt_util_log)
print(f'MSE for Optimized log Lasso Model: {mse_lasso_opt_util_log:.2f}')
print(f'R^2 for Optimized log Lasso Model: {r2_lasso_opt_util_log * 100:.2f}%')





# I am done






