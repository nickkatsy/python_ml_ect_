import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import minimize
import numpy as np
import warnings
warnings.filterwarnings('ignore')



# applications, EDA, cleaning ecttt

applications = pd.read_csv('https://raw.githubusercontent.com/LeeMorinUCF/QMB6358F23/main/final_exam_2021/applications.csv')
applications.info()
applications.isna().sum()
applications.nunique()
print(applications.dtypes)
print(applications.describe())

df_app = applications.copy()
df_app = df_app.drop(['app_id','ssn','zip_code'],axis=1)

df_app['homeownership'] = pd.get_dummies(df_app.homeownership,prefix='homeownership').iloc[:,0:1]
df_app.isna().sum()
df_app.describe()

#heatmap to show correations amoung features of the set
sns.heatmap(df_app.corr(),annot=True)
plt.show()



X = df_app.drop('purchases',axis=1)
y = df_app['purchases']

import statsmodels.api as sm

model1 = sm.OLS(exog=X,endog=y).fit()
print(model1.summary())

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

from sklearn.linear_model import Ridge,Lasso
lasso = Lasso()
ridge = Ridge()


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor()



from sklearn.metrics import r2_score,mean_squared_error

def evaluate_add(model,X_train_scaled,X_test_scaled,y_train,y_test):
    model = model.fit(X_train_scaled,y_train)
    pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test,pred)
    mse = mean_squared_error(y_test,pred)
    print(f'{model.__class__.__name__}, -R2 Score- {r2*100:.2f}%; --MSE-- {mse/80:.2f}')
    return pred


lr_pred = evaluate_add(lr,X_train_scaled,X_test_scaled,y_train,y_test)
rfr_pred = evaluate_add(rfr,X_train_scaled,X_test_scaled,y_train,y_test)
tree_pred = evaluate_add(tree,X_train_scaled,X_test_scaled,y_train,y_test)
ridge_pred = evaluate_add(ridge,X_train_scaled,X_test_scaled,y_train,y_test)
lasso_pred = evaluate_add(lasso,X_train_scaled,X_test_scaled,y_train,y_test)

#import credit dataset

credit = pd.read_csv('https://raw.githubusercontent.com/LeeMorinUCF/QMB6358F23/main/final_exam_2021/credit_bureau.csv')
print(credit.info())
credit.isna().sum()
credit.drop_duplicates(inplace=True)
print(credit.dtypes)
credit.describe()
df_credit = credit.copy()
df_credit.isna().sum()

#heatmap for credit
sns.heatmap(df_credit.corr(),annot=True)
plt.show()

#merging datasets to make purch_app.csv dataset

purch_app = pd.concat([df_app,df_credit],axis=1,join='inner')
purch_app.isna().sum()
purch_app.drop_duplicates(inplace=True)


purch_app.isna().sum()

print(purch_app.describe())

# using minmax scaler on a different version of purch_app to scale the datafram through minmaxscaler

from sklearn.preprocessing import MinMaxScaler

ms = MinMaxScaler()

df_purch = ms.fit_transform(purch_app)

#predict purchases as a function of the other varaibles, ignore 'app_id','ssn','zip_code'

X_purch = purch_app.drop('purchases',axis=1)
y_purch = purch_app['purchases']

model_purch1 = sm.OLS(exog=X_purch,endog=y_purch).fit()
print(model_purch1.summary())

X_purch = purch_app.drop('purchases',axis=1)
y_purch = purch_app['purchases']



model_purch1 = sm.OLS(exog=X_purch,endog=y_purch).fit()
print(model_purch1.summary())

from sklearn.model_selection import train_test_split
X_purch_train,X_purch_test,y_purch_train,y_purch_test = train_test_split(X_purch,y_purch,test_size=.20,random_state=42)


from sklearn.linear_model import LinearRegression

lr_purch = LinearRegression()


from sklearn.linear_model import Lasso,Ridge

lasso_purch = Lasso()
ridge_purch = Ridge()

from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
rfr_purch = RandomForestRegressor()
gbr_purch = GradientBoostingRegressor()

# metrics from the purch_app combined dataframe

def evaluate_purch(model,X_purch_train,X_purch_test,y_purch_train,y_purch_test):
    model.fit(X_purch_train, y_purch_train)
    pred = model.predict(X_purch_test)
    r2_purch = r2_score(y_purch_test,pred)
    mse_purch = mean_squared_error(y_purch_test,pred)
    print(f'{model.__class__.__name__}, --R2-- {r2_purch*100:.2f}%; --MSE-- {mse_purch/80:.2f}')
    return pred


lr_purch_pred = evaluate_purch(lr_purch, X_purch_train, X_purch_test, y_purch_train, y_purch_test)
lasso_purch_pred = evaluate_purch(lasso_purch,X_purch_train,X_purch_test,y_purch_train,y_purch_test)
ridge_purch_pred = evaluate_purch(ridge_purch,X_purch_train,X_purch_test,y_purch_train,y_purch_test)
gbr_purch_pred = evaluate_purch(gbr_purch,X_purch_train,X_purch_test,y_purch_train,y_purch_test)
rfr_purch_pred = evaluate_purch(rfr_purch,X_purch_train,X_purch_test,y_purch_train,y_purch_test)



#importing demographics dataset and then same old stuff

demographic = pd.read_csv('https://raw.githubusercontent.com/LeeMorinUCF/QMB6358F23/main/final_exam_2021/demographic.csv')

demographic.info()
demographic.isna().sum()
print(demographic.dtypes)

df_dem = demographic.copy()

#heatmap for correlations from dem dataset

sns.heatmap(df_dem.corr(),annot=True)
plt.show()



# Merging datasets to make purch_full dataframe
purch_full = pd.concat([df_dem, purch_app], axis=1, join='inner')
purch_full.drop_duplicates(inplace=True)

#purch_full = purch_full.drop('zip_code',axis=1)
purch_full.isna().sum()
purch_full.describe()

#drop ssn,zid_code, 

purch_full = purch_full.drop('zip_code',axis=1)
purch_full = purch_full.drop('ssn',axis=1)
X_full = purch_full.drop('purchases',axis=1)
y_full = purch_full['purchases']

model2 = sm.OLS(exog=X_purch,endog=y_purch).fit()
model2.summary()


from sklearn.model_selection import train_test_split
X_train_full,X_test_full,y_train_full,y_test_full = train_test_split(X_full,y_full,test_size=.20,random_state=42)

from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler()

X_train_full_scaled = ms.fit_transform(X_train_full)
X_test_full_scaled = ms.transform(X_test_full)

from sklearn.linear_model import LinearRegression,Ridge,Lasso
lr_full = LinearRegression()
ridge_full = Ridge()
lasso_full = Lasso()

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
rfr_full = RandomForestRegressor()
gbr_full = GradientBoostingRegressor()

def evaluate_full(model,X_train_full_scaled,X_test_full_scaled,y_train_full,y_test_full):
    model = model.fit(X_train_full_scaled, y_train_full)
    pred = model.predict(X_test_full_scaled)
    r2_full = r2_score(y_test_full,pred)
    mse_full = mean_squared_error(y_test_full,pred)
    print(f'{model.__class__.__name__}, --R2-- {r2_full*100:.2f}%; --MSE-- {mse_full*0.05:.2f}')
    return pred

lr_full_pred = evaluate_full(lr_full, X_train_full_scaled, X_test_full_scaled, y_train_full, y_test_full)
ridge_full_pred = evaluate_full(ridge_full,X_train_full_scaled,X_test_full_scaled,y_train_full,y_test_full)
lasso_full_scaled = evaluate_full(lasso_full,X_train_full_scaled,X_test_full_scaled,y_train_full,y_test_full)
rfr_full_scaled = evaluate_full(rfr_full,X_train_full_scaled,X_test_full_scaled,y_train_full,y_test_full)
gbr_full = evaluate_full(gbr_full,X_train_full_scaled,X_test_full_scaled,y_train_full,y_test_full)

#creating utlity function


utilitilzation = purch_full['purchases'] / purch_full['credit_limit']

print(utilitilzation.describe())

# making uility the dependent variable
y_util = utilitilzation

model3 = sm.OLS(exog=purch_full,endog=y_util).fit()
print(model3.summary())


from sklearn.preprocessing import train_test_split
X_train_util,X_test_util,y_train_util,y_test_util = train_test_split(X_full,y_util,test_size=.20,random_state=42)


from sklearn.linear_model import LinearRegression,Ridge,Lasso

lr_util = LinearRegression()
ridge_util = Ridge()
lasso_util = Lasso()
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
rfr_util = RandomForestRegressor()
gbr_util = GradientBoostingRegressor()



def evaluate_util(model,X_train_util,X_test_util,y_train_util,y_test_util):
    model = model.fit(X_train_util, y_train_util)
    pred = model.predict(X_test_util)
    r2_util = r2_score(y_test_util,pred)
    mse_util = mean_squared_error(y_test_util,pred)
    print(f'{model.__class__.__name__}, --R2-- {r2_util*100:.2f}%; --MSE-- {mse_util*0.05:.2f}')
    return pred

lr_util_pred = evaluate_util(lr_util,X_train_util,X_test_util,y_train_util,y_test_util)
ridge_util_pred = evaluate_util(ridge_util,X_train_util,X_test_util,y_train_util,y_test_util)
lasso_util_pred = evaluate_util(lasso_util,X_train_util,X_test_util,y_train_util,y_test_util)
rfr_util_pred = evaluate_util(rfr_util,X_train_util,X_test_util,y_train_util,y_test_util)
gbr_util_pred = evaluate_util(gbr_util,X_train_util,X_test_util,y_train_util,y_test_util)





# creating logs_odd_util

import numpy as np


log_util = np.log(utilitilzation) / 1 - utilitilzation

log_util.describe()

log_util.isna().sum()

y_log = log_util

model_log = sm.OLS(exog=X_full,endog=y_log).fit()
print(model_log.summary())



#optimization time

from scipy.optimize import minimize

def SSR(beta,X,y):
    pred = np.dot(X,beta)
    residuals = y - pred
    ssr = np.sum((residuals)**2)
    return ssr



def Random_Forest_SSR(beta,model,X,y):
    pred = model.predict(X)
    residuals = y - pred
    ssr = np.sum(residuals**2)
    return ssr

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


#will finisih the next day too tired


