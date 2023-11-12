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



sns.scatterplot(copy,x='area',y='price')


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


def violin(copy):
    dis_,axs__ = plt.subplots(2,2,figsize=(10,6))
    sns.violinplot(copy[['bedrooms']],ax=axs__[0,0])
    sns.violinplot(copy[['stories']],ax=axs__[0,1])
    sns.violinplot(copy[['bathrooms']],ax=axs__[1,0])
    sns.violinplot(copy[['basement']],ax=axs__[1,1])
    plt.show()
    

violin(copy)







X = df.drop('price',axis=1)
y = df.price

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)



from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor



lr = LinearRegression()
rf = RandomForestRegressor()
dt = DecisionTreeRegressor()



from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import make_column_transformer

ohe = OneHotEncoder()

ct = make_column_transformer(
    (ohe,['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea','furnishingstatus']
     ),remainder='passthrough')



ct.fit_transform(X)


from sklearn.pipeline import make_pipeline

pipe_lr = make_pipeline(ct,lr).fit(X_train,y_train)

pipe_rf = make_pipeline(ct,rf).fit(X_train,y_train)

pipe_tree = make_pipeline(ct,dt).fit(X_train,y_train)



lr_score = pipe_lr.predict(X_test)

rf_score = pipe_rf.predict(X_test)

tree_score = pipe_tree.predict(X_test)




from sklearn.metrics import r2_score,mean_squared_error

#Linear Regression Results

print('R2 for linear Regression model= ',r2_score(y_test, lr_score)*100)

print('MSE for linear Regression Model= ',mean_squared_error(y_test, lr_score)*100)

#Random Forrest Regression Results

print('R2 value for Random Forest Regression= ',r2_score(y_test, rf_score)*100)

print('MSE for Random Forest Regression Model =',mean_squared_error(y_test, rf_score)*100)

# Decision Tree Results

print('R2 for Decision Tree= ', r2_score(y_test, tree_score)*100)

print('MSE for Decision Tree=  ', mean_squared_error(y_test, tree_score)*100)
