import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

url = 'https://raw.githubusercontent.com/nickkas/python_ml_ect_/master/Housing.csv'

df = pd.read_csv(url)


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



bar, ax1 = plt.subplots(2,3,figsize=(10,6))
bar1 = sns.barplot(copy,x='bedrooms',y='price',ax = ax1[0,0])
bar2 = sns.barplot(copy,x='stories',y='price',ax= ax1[0,1])
bar3 = sns.barplot(copy,x='bathrooms',y='price',ax = ax1[0,2])
bar4 = sns.barplot(copy,x='parking',y='price',ax = ax1[1,0])
bar5 = sns.barplot(copy,x='furnishingstatus',y='price',ax=ax1[1,1])
bar6 = sns.barplot(copy,x='airconditioning',y='price',ax=ax1[1,2])


box0, ax2 = plt.subplots(2,3,figsize=(10,6))
box1 = sns.boxplot(copy[['area']],ax=ax2[0,0])
box2 = sns.boxplot(copy['price'],ax=ax2[0,1])
box3 = sns.boxplot(copy['parking'],ax=ax2[0,2])
box4 = sns.boxplot(copy['stories'],ax=ax2[1,0])
box5 = sns.boxplot(copy['bedrooms'],ax=ax2[1,1])
box6 = sns.boxplot(copy['bathrooms'],ax=ax2[1,2])


plt.figure(figsize=(10,6))
displot1 = sns.displot(copy[['bedrooms']])
displot2 = sns.displot(copy[['stories']])
displot3 = sns.displot(copy[['bathrooms']])
displot4 = sns.displot(copy[['basement']])




plt.figure(figsize=(10,6))
sns.kdeplot(copy,x='price')
sns.kdeplot(copy,x='area')



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

print('R2 for linear Regression model= ',r2_score(y_test, lr_score)*100)

print('MSE for linear Regression Model= ',mean_squared_error(y_test, lr_score)*100)


print('R2 value for Random Forrest Regression= ',r2_score(y_test, rf_score)*100)

print('MSE for Random Forrest Regression Model =',mean_squared_error(y_test, rf_score)*100)


print('R2 for Decesion Tree= ', r2_score(y_test, tree_score))

print('MSE for Decesion tree= ', mean_squared_error(y_test, tree_score))
