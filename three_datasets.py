import pandas as pd
import warnings
warnings.filterwarnings('ignore')

applications = pd.read_csv('https://raw.githubusercontent.com/LeeMorinUCF/QMB6358F23/main/final_exam_2021/applications.csv')

applications.info()
applications.describe()
applications.isna().sum()
applications.drop_duplicates(inplace=True)

df_app = applications.copy()

#drop ssn,app_id, and zip_code

df_app = df_app.drop(['ssn','zip_code','app_id'],axis=1)

df_app['homeownership'] = pd.get_dummies(df_app.homeownership,prefix='homeownership').iloc[:,0:1]

X_purch = df_app.drop('purchases',axis=1)
y_purch = df_app['purchases']


import statsmodels.api as sm
model1 = sm.OLS(exog=X_purch,endog=y_purch).fit()
print(model1.summary())

# train/test split model evaluation


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X_purch,y_purch,test_size=.20,random_state=42)


from sklearn.preprocessing import MinMaxScaler

ms = MinMaxScaler()

X_train_scaled = ms.fit_transform(X_train)
X_test_scaled = ms.transform(X_test)

from sklearn.linear_model import LinearRegression, Lasso,Ridge
lr = LinearRegression()
ridge = Ridge()
lasso = Lasso()


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

rfr = RandomForestRegressor()
gbr = GradientBoostingRegressor()

from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor()



from sklearn.metrics import r2_score,mean_squared_error

def evaluate_first_model(model,X_train_scaled,X_test_scaled,y_train,y_test):
    model = model.fit(X_train_scaled,y_train)
    pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, pred)
    mse = mean_squared_error(y_test,pred)
    print(f'{model.__class__.__name__}, --R2-- {r2*100:.2f}%; --MSE-- {mse/80:.2f}')
    return pred

lr_pred = evaluate_first_model(lr,X_train_scaled, X_test_scaled, y_train, y_test)
rfr_pred = evaluate_first_model(rfr, X_train_scaled, X_test_scaled, y_train, y_test)
gbr_pred = evaluate_first_model(gbr, X_train_scaled, X_test_scaled, y_train, y_test)
tree_pred = evaluate_first_model(tree, X_train_scaled, X_test_scaled, y_train, y_test)
ridge_pred = evaluate_first_model(ridge, X_train_scaled, X_test_scaled, y_train, y_test)
lasso_pred = evaluate_first_model(lasso, X_train_scaled, X_test_scaled, y_train, y_test)


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

X = df_purch.drop('purchases',axis=1)
y = df_purch['purchases']

model2 = sm.OLS(exog=X,endog=y).fit()
print(model2.summary())



lr_purch = LinearRegression()
ridge_purch = Ridge()
lasso_purch = Lasso()


from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
rfr_purch = RandomForestRegressor()
gbr_purch = GradientBoostingRegressor()

from sklearn.tree import DecisionTreeRegressor
tree_purch = DecisionTreeRegressor()



X_train_purch,X_test_purch,y_train_purch,y_test_purch = train_test_split(X,y,test_size=.20,random_state=42)


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
tree_purch_pred = evaluate_purch(tree_purch, X_train_purch_scaled, X_test_purch_scaled, y_train_purch, y_test_purch)



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
    sns.scatterplot(x='credit_limit',y='avg_income',ax=axs[0,0],data=purch_full)
    axs[0,0].set_title('Credit Limit VS Average Income')
    
    
    sns.violinplot(x='num_late',y='homeownership',ax=axs[0,1],data=purch_full)
    axs[0,1].set_title('Numer of Late payments vs Homeownership')
    
    
    sns.barplot(x='num_bankruptcy',y='credit_limit',ax=axs[1,0],data=purch_full)
    axs[1,0].set_title('Credit limit vs Number of Individuals who filed for BankRuptcy')
    
    
    sns.scatterplot(x='purchases',y='income',ax=axs[1,1],data=purch_full)
    
    plt.show()


desc_full(purch_full)








X_full = purch_full.drop('purchases',axis=1)
y_full = purch_full['purchases']

model_full = sm.OLS(exog=X_full,endog=y_full).fit()
print(model_full.summary())


X_full_train,X_full_test,y_full_train,y_full_test = train_test_split(X_full,y_full,test_size=.20,random_state=42)





lr_full = LinearRegression()
ridge_full = Ridge()
lasso_full = Lasso()



rfr_full = RandomForestRegressor()
gbr_full = GradientBoostingRegressor()


tree_full = DecisionTreeRegressor()





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
tree_full_pred = full_model(tree_full, X_full_train, X_full_test, y_full_train, y_full_test)




#create utilization variable

utilization = purch_full['purchases'] / purch_full['credit_limit']

utilization.info()
utilization.isna().sum()
utilization.describe()


X_util = purch_full
y_util = utilization

model_util = sm.OLS(exog=X_util,endog=y_util).fit()
print(model_util.summary())


X_util_train,X_util_test,y_util_train,y_util_test = train_test_split(X_util,y_util,test_size=.20,random_state=42)



lr_util = LinearRegression()
ridge_util = Ridge()
lasso_util = Lasso()


rfr_util = RandomForestRegressor()
gbr_util = GradientBoostingRegressor()

tree_util = DecisionTreeRegressor()


def evaluate_util(model_util,X_util_train,X_util_test,y_util_train,y_util_test):
    model_util = model_util.fit(X_util_train,y_util_train)
    pred = model_util.predict(X_util_test)
    r2_util = r2_score(y_util_test, pred)
    mse_util = mean_squared_error(y_util_test,pred)
    print(f'{model_util.__class__.__name__}, --R2== {r2_util*100:.2f}%; --MSE-- {mse_util:.2f}')
    return pred


lr_util_pred = evaluate_util(lr_util, X_util_train, X_util_test, y_util_train, y_util_test)
ridge_util_pred = evaluate_util(ridge_util, X_util_train, X_util_test, y_util_train, y_util_test)
lasso_util_pred = evaluate_util(lasso_util, X_util_train, X_util_test, y_util_train, y_util_test)
ridge_util_pred = evaluate_util(ridge_util, X_util_train, X_util_test, y_util_train, y_util_test)
rfr_util_pred = evaluate_util(rfr_util, X_util_train, X_util_test, y_util_train, y_util_test)
gbr_util_pred = evaluate_util(gbr_util, X_util_train, X_util_test, y_util_train, y_util_test)
tree_util_pred = evaluate_util(tree_util, X_util_train, X_util_test, y_util_train, y_util_test)



import numpy as np

log_odds_util = np.log(utilization) / 1 - utilization

y_log_util = log_odds_util


lr_util_log = LinearRegression()

ridge_util_log = Ridge()
lasso_util_log = Lasso()
rfr_util_log = RandomForestRegressor()
gbr_util_log = GradientBoostingRegressor()
tree_util_log = DecisionTreeRegressor()
ridge_util_log = Ridge()

X_util_log_train,X_util_log_test,y_util_log_train,y_util_log_test = train_test_split(X_util,y_log_util,test_size=.20,random_state=42)


def evaluate_log_util(model_util_log,X_util_log_train,X_util_log_test,y_util_log_train,y_util_log_test):
    model_util_log = model_util_log.fit(X_util_log_train,y_util_log_train)
    pred = model_util_log.predict(X_util_log_test)
    r2_util_log = r2_score(y_util_log_test, pred)
    mse_util_log = mean_squared_error(y_util_log_test,pred)
    print(f'{model_util_log.__class__.__name__}, --R2== {r2_util_log*100:.2f}%; --MSE-- {mse_util_log:.2f}')
    return pred


lr_util_log_pred = evaluate_log_util(lr_util_log, X_util_log_train, X_util_log_test, y_util_log_train, y_util_log_test) 
lasso_util_log = evaluate_log_util(lasso_util_log, X_util_log_train, X_util_log_test, y_util_log_train, y_util_log_test)
rfr_util_log = evaluate_log_util(rfr_util_log, X_util_log_train, X_util_log_test, y_util_log_train, y_util_log_test)
gbr_util_log = evaluate_log_util(gbr_util_log, X_util_log_train, X_util_log_test, y_util_log_train, y_util_log_test)
tree_util_log = evaluate_log_util(tree_util_log, X_util_log_train, X_util_log_test, y_util_log_train, y_util_log_test)
ridge_util_log_pred = evaluate_log_util(ridge_util_log, X_util_log_train, X_util_log_test, y_util_log_train, y_util_log_test)




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


# testing SSR Function
initial_guess = np.zeros(X_full.shape[1] + 1)  

result = minimize(SSR_Linear_Regression,x0=initial_guess,args=(X_full.values,y_full.values),method='Nelder-Mead')
optimal_params = result.x

# Extracting coefficients and intercept from optimal_params
optimal_intercept = optimal_params[0]
optimal_coefficients = optimal_params[1:]

# optimal model from 'Full'
optimal_model = LinearRegression()
optimal_model.intercept_ = optimal_intercept
optimal_model.coef_ = optimal_coefficients

# 'Full' model
y_pred_optimal = optimal_model.predict(X_full.values)
mse_optimal = mean_squared_error(y_full.values,y_pred_optimal)
r2_optimal = r2_score(y_full.values,y_pred_optimal)

print(f'Optimal Coefficients: {optimal_coefficients}')
print(f'MSE for Optimized Model: {mse_optimal}')
print(f'R^2 for Optimized Model: {r2_optimal}')




#linear regression for utility
initial_guess = np.zeros(X_util.shape[1] + 1)  

result = minimize(SSR_Linear_Regression,x0=initial_guess,args=(X_util.values,y_util.values),method='Nelder-Mead')
optimal_params = result.x


optimal_intercept = optimal_params[0]
optimal_coefficients = optimal_params[1:]


optimal_model = LinearRegression()
optimal_model.intercept_ = optimal_intercept
optimal_model.coef_ = optimal_coefficients


y_pred_optimal = optimal_model.predict(X_util.values)
mse_optimal = mean_squared_error(y_util.values, y_pred_optimal)
r2_optimal = r2_score(y_util.values, y_pred_optimal)

print(f'Optimal Coefficients: {optimal_coefficients}')
print(f'MSE for Optimized Model: {mse_optimal}')
print(f'R^2 for Optimized Model: {r2_optimal}')


# Linear Regression for purch

initial_guess = np.zeros(X_purch.shape[1] + 1)

result = minimize(SSR_Linear_Regression,x0=initial_guess,args=(X_purch.values,y_purch.values),method='Nelder-Mead')
optimal_params = result.x


optimal_intercept = optimal_params[0]
optimal_coefficients = optimal_params[1:]


optimal_model = LinearRegression()
optimal_model.intercept_ = optimal_intercept
optimal_model.coef_ = optimal_coefficients


y_pred_optimal = optimal_model.predict(X_purch.values)
mse_optimal = mean_squared_error(y_purch.values, y_pred_optimal)
r2_optimal = r2_score(y_purch.values, y_pred_optimal)

print(f'Optimal Coefficients: {optimal_coefficients}')
print(f'MSE for Optimized Model: {mse_optimal}')
print(f'R^2 for Optimized Model: {r2_optimal}')



#linear regression log utility

initial_guess = np.zeros(X_util.shape[1] + 1)

result = minimize(SSR_Linear_Regression,x0=initial_guess,args=(X_util.values,y_log_util.values),method='Nelder-Mead')
optimal_params = result.x


optimal_intercept = optimal_params[0]
optimal_coefficients = optimal_params[1:]


optimal_model = LinearRegression()
optimal_model.intercept_ = optimal_intercept
optimal_model.coef_ = optimal_coefficients


y_pred_optimal = optimal_model.predict(X_util.values)
mse_optimal = mean_squared_error(y_log_util.values,y_pred_optimal)
r2_optimal = r2_score(y_log_util.values,y_pred_optimal)

print(f'Optimal Coefficients: {optimal_coefficients}')
print(f'MSE for Optimized Model: {mse_optimal}')
print(f'R^2 for Optimized Model: {r2_optimal}')


#random forest

def Random_Forest_SSR(beta,model,X,y):
    pred = model.predict(X)
    residuals = y - pred
    ssr = np.sum(residuals**2)
    return ssr


#rfr full

model_rfr_full = RandomForestRegressor()
model_rfr_full.fit(X_full,y_full)




initial_beta_rfr_full = model_rfr_full.feature_importances_
ssr_from_rfr_full = Random_Forest_SSR(initial_beta_rfr_full,model_rfr_full,X_full.values,y_full.values)
ssr_from_rfr_full_t = np.sum(model_rfr_full.predict(X_full.values) - (y_full.values*2))
print(f'SSR from function: {ssr_from_rfr_full}')
print(f'SSR from model: {ssr_from_rfr_full_t}')


#initial guess for rfr_full

initial_guess_rfr_full = np.zeros(X_full.shape[1])
result_rfr_full = minimize(Random_Forest_SSR,x0=initial_guess_rfr_full,args=(model_rfr_full,X_full.values,y_full.values),method='Nelder-Mead')
optimal_params_rfr_full = result_rfr_full.x


# predictions of optimal params on training set
y_pred_rfr_full = model_rfr_full.predict(X_full.values)
mse_rfr_full = mean_squared_error(y_full.values,y_pred_rfr_full)
r2_rfr_full = r2_score(y_full.values,y_pred_rfr_full)
print(f'Mean Squared Error and R2 values from optimal params on training data: --MSE-- {mse_rfr_full/.05:.2f} --R2-- {r2_rfr_full*100:.2f}%')



# rfr utility

model_rfr_util = RandomForestRegressor()
model_rfr_util.fit(X_full,y_util)




initial_beta_rfr_util = model_rfr_util.feature_importances_
ssr_from_rfr_util = Random_Forest_SSR(initial_beta_rfr_util,model_rfr_util,X_full.values,y_util.values)
ssr_from_rfr_util_t = np.sum(model_rfr_util.predict(X_full.values) - (y_util.values*2))
print(f'SSR from function: {ssr_from_rfr_util}')
print(f'SSR from model: {ssr_from_rfr_util_t}')


#initial guess for rfr_util

initial_guess_rfr_util = np.zeros(X_full.shape[1])
result_rfr_util = minimize(Random_Forest_SSR,x0=initial_guess_rfr_util,args=(model_rfr_util,X_full.values,y_util.values),method='Nelder-Mead')
optimal_params_rfr_util = result_rfr_util.x


# predictions of optimal params on training set
y_pred_rfr_util = model_rfr_util.predict(X_full.values)
mse_rfr_util = mean_squared_error(y_util.values,y_pred_rfr_util)
r2_rfr_util = r2_score(y_util.values,y_pred_rfr_util)
print(f'Mean Squared Error and R2 values from optimal params on training data: --MSE-- {mse_rfr_util/.05:.2f} --R2-- {r2_rfr_util*100:.2f}%')


#Random Forrest purch

model_rfr_purch = RandomForestRegressor()
model_rfr_purch.fit(X_purch,y_purch)




initial_beta_rfr_purch = model_rfr_purch.feature_importances_
ssr_from_rfr_purch = Random_Forest_SSR(initial_beta_rfr_purch,model_rfr_purch,X_purch.values,y_purch.values)
ssr_from_rfr_purch_t = np.sum(model_rfr_purch.predict(X_purch.values) - (y_purch.values*2))
print(f'SSR from function: {ssr_from_rfr_purch}')
print(f'SSR from model: {ssr_from_rfr_purch_t}')


#initial guess for rfr_purch

initial_guess_rfr_purch = np.zeros(X_purch.shape[1])
result_rfr_purch = minimize(Random_Forest_SSR,x0=initial_guess_rfr_purch,args=(model_rfr_purch,X_purch.values,y_purch.values),method='Nelder-Mead')
optimal_params_rfr_purch = result_rfr_purch.x


# predictions of optimal params on training set
y_pred_rfr_purch = model_rfr_purch.predict(X_purch.values)
mse_rfr_purch = mean_squared_error(y_purch.values,y_pred_rfr_purch)
r2_rfr_purch = r2_score(y_purch.values,y_pred_rfr_purch)
print(f'Mean Squared Error and R2 values from optimal params on training data: --MSE-- {mse_rfr_purch/.05:.2f} --R2-- {r2_rfr_purch*100:.2f}%')





#lasso full

initial_guess_lasso = np.zeros(X_full.shape[1])


def SSR_lasso(beta,X,y,alpha):
    model = Lasso(alpha=alpha)
    model.fit(X,y)
    pred = model.predict(X)
    residuals = y - pred
    ssr = np.sum(residuals**2) + alpha * np.sum(np.abs(beta))  
    return ssr


alpha_lasso = 0.1
result_lasso = minimize(SSR_lasso,x0=initial_guess_lasso,args=(X_full.values,y_full.values,alpha_lasso),method='Nelder-Mead')
optimal_params_lasso = result_lasso.x
print(f'Optimal Parameters for Lasso Regression: {optimal_params_lasso}')


lasso_model = Lasso(alpha=alpha_lasso)
lasso_model.fit(X_full, y_full)

# Test the Lasso model performance
y_pred_lasso = lasso_model.predict(X_full)
r2_lasso = r2_score(y_full,y_pred_lasso)
mse_lasso = mean_squared_error(y_full,y_pred_lasso)

print(f'Optimized Lasso Regression, --R2-- {r2_lasso*100:.2f}%; --MSE-- {mse_lasso:.2f}')


# lasso utility


initial_guess_lasso = np.zeros(X_util.shape[1])

alpha_lasso = 0.1
result_lasso = minimize(SSR_lasso, x0=initial_guess_lasso, args=(X_util.values,y_util.values,alpha_lasso), method='Nelder-Mead')
optimal_params_lasso = result_lasso.x
print(f'Optimal Parameters for Lasso Regression: {optimal_params_lasso}')

# Update the Lasso model with optimal parameters
lasso_model = Lasso(alpha=alpha_lasso)
lasso_model.fit(X_util, y_util)

# Test the Lasso model performance
y_pred_lasso = lasso_model.predict(X_util)
r2_lasso = r2_score(y_util, y_pred_lasso)
mse_lasso = mean_squared_error(y_util, y_pred_lasso)

print(f'Optimized Lasso Regression, --R2-- {r2_lasso*100:.2f}%; --MSE-- {mse_lasso:.2f}')













