import pandas as pd
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('C:/ML/python/data/housing.csv',delimiter=',')

df.info()
df.dtypes
df.nunique()
df['furnishingstatus'].value_counts()


def turn_dummies_into_numeric(df):
    df['mainroad'] = [1 if X == 'yes' else 0 for X in df['mainroad']]
    df['guestroom'] = [1 if X == 'yes' else 0 for X in df['guestroom']]

    df['basement'] = [1 if X == 'yes' else 0 for X in df['basement']]
    df['hotwaterheating'] = [1 if X == 'yes' else 0 for X in df['hotwaterheating']]
    df['airconditioning'] = [1 if X == 'yes' else 0 for X in df['airconditioning']]

    df['prefarea'] = [1 if X == 'yes' else 0 for X in df['prefarea']]
    return df


turn_dummies_into_numeric(df)

furnish_mapping = {'semi-furnished':1,'unfurnished':2,'furnished':3}

df['furnishingstatus'] = df['furnishingstatus'].map(furnish_mapping)


import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(df.corr(),annot=True)
plt.show()


def desc_stats(df):
    f,axs = plt.subplots(2,2,figsize=(10,6))
    sns.boxplot(x='mainroad',y='price',ax=axs[0,0],data=df)
    sns.scatterplot(x='area',y='price',ax=axs[0,1],data=df)
    sns.barplot(x='parking',y='stories',ax=axs[1,0],data=df)
    sns.boxplot(x='basement',y='price',ax=axs[1,1],data=df)
    plt.show()



desc_stats(df)


X = df.drop('price',axis=1)
y = df['price']

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)

X_train_scaled = poly.fit_transform(X_train)
X_test_scaled = poly.transform(X_test)


from sklearn.linear_model import LinearRegression,Ridge,Lasso

lr = LinearRegression()
ridge = Ridge(alpha=.8)
lasso = Lasso(alpha=.8)

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,BaggingRegressor
RFR = RandomForestRegressor()
BR = BaggingRegressor()
GBR = GradientBoostingRegressor()


from sklearn.metrics import mean_squared_error,r2_score


def evaluation(X_train_scaled,X_test_scaled,y_train,y_test,model):
    model = model.fit(X_train_scaled,y_train)
    pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test,pred)
    mse = mean_squared_error(y_test,pred)
    print(f'{model.__class__.__name__}, --R2 Score-- {r2*100:.2f}%; --MSE-- {mse:.2f}')
    return pred

lr_pred = evaluation(X_train_scaled, X_test_scaled, y_train, y_test, lr)
GBR_pred = evaluation(X_train_scaled, X_test_scaled, y_train, y_test, GBR)
lasso_pred = evaluation(X_train_scaled, X_test_scaled, y_train, y_test, lasso)
ridge_pred = evaluation(X_train_scaled, X_test_scaled, y_train, y_test, ridge)
RFR_pred = evaluation(X_train_scaled, X_test_scaled, y_train, y_test, RFR)
BR_pred = evaluation(X_train_scaled, X_test_scaled, y_train, y_test, BR)
