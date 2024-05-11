import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')



df = pd.read_csv('https://raw.githubusercontent.com/nickkas/python_ml_ect_/master/Housing.csv')


df.info()
df.isna().sum()
df.columns
df.nunique()
df.dtypes


copy = df.copy()


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



for i in copy:
    copy[i] = le.fit_transform(copy[i])



copy.corr()


def scatter(copy):
    sns.scatterplot(copy,x='area',y='price')
    plt.plot()


scatter(copy)




def bar(copy):
    bar, ax1 = plt.subplots(2,3,figsize=(10,6))
    sns.barplot(copy,x='bedrooms',y='price',ax = ax1[0,0])
    sns.barplot(copy,x='stories',y='price',ax= ax1[0,1])
    sns.barplot(copy,x='bathrooms',y='price',ax = ax1[0,2])
    sns.barplot(copy,x='parking',y='price',ax = ax1[1,0])
    sns.barplot(copy,x='furnishingstatus',y='price',ax=ax1[1,1])
    sns.barplot(copy,x='airconditioning',y='price',ax=ax1[1,2])
    plt.show()
    
    
bar(copy)

def box(copy):
    box0, ax2 = plt.subplots(2,3,figsize=(10,6))
    sns.boxplot(copy[['area']],ax=ax2[0,0])
    sns.boxplot(copy['price'],ax=ax2[0,1])
    sns.boxplot(copy['parking'],ax=ax2[0,2])
    sns.boxplot(copy['stories'],ax=ax2[1,0])
    sns.boxplot(copy['bedrooms'],ax=ax2[1,1])
    sns.boxplot(copy['bathrooms'],ax=ax2[1,2])
    plt.show()
    
box(copy)


def subs(copy):
    dis_,axs__ = plt.subplots(2,2,figsize=(10,6))
    sns.barplot(x='bedrooms',y='price',ax=axs__[0,0],data=copy)
    sns.boxplot(x='stories',y='price',ax=axs__[0,1],data=copy)
    sns.barplot(x='mainroad',y='price',ax=axs__[1,0],data=copy)
    sns.scatterplot(x='area',y='price',ax=axs__[1,1],data=copy)
    plt.show()
    

subs(copy)



X = df.drop('price',axis=1)
y = df.price

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)



from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

ridge = Ridge(alpha=.7)
lasso = Lasso(alpha=.8)


lr = LinearRegression()
RFR = RandomForestRegressor()
Tree = DecisionTreeRegressor()
GBR = GradientBoostingRegressor()
BR = BaggingRegressor()


from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
ms = MinMaxScaler()



ohe = OneHotEncoder()





from sklearn.compose import make_column_transformer


ct = make_column_transformer(
    (ohe,X.select_dtypes(include='object').columns),
     (ms,X.select_dtypes(include=['float64','int64']).columns),remainder='passthrough')

ct.fit_transform(X)


from sklearn.pipeline import make_pipeline






from sklearn.metrics import r2_score,mean_squared_error

def evaluate_model(X_train,X_test,y_train,y_test,model):
    pipe = make_pipeline(ct,model).fit(X_train,y_train)
    pred = pipe.predict(X_test)
    mse = mean_squared_error(y_test,pred)
    r2 = r2_score(y_test,pred)
    print(f'{model.__class__.__name__}, --r2-- {r2*100:.2f}%; --mse-- {mse:.2f}')
    return pred

lr_pred = evaluate_model(X_train, X_test, y_train, y_test, lr)
GBR_pred = evaluate_model(X_train, X_test, y_train, y_test, GBR)
RFR_pred = evaluate_model(X_train, X_test, y_train, y_test, RFR)
Tree_pred = evaluate_model(X_train, X_test, y_train, y_test, Tree)
lasso_pred = evaluate_model(X_train, X_test, y_train, y_test, lasso)
ridge_pred = evaluate_model(X_train, X_test, y_train, y_test, ridge)
BR_pred = evaluate_model(X_train, X_test, y_train, y_test, BR)
