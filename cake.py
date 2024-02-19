import pandas as pd
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('https://raw.githubusercontent.com/LeeMorinUCF/QMB6912S23/main/demo_11/FlyReel_Paper/Data/FlyReels.csv')
df.info()
df.isna().sum()
df.describe()
print(df.dtypes)
df.nunique()

df.columns = df.columns.str.replace(' ','_')
df1 = df.copy()


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()


for i in df1:
    df1[i] = le.fit_transform(df1[i])
    
    


import matplotlib.pyplot as plt
import seaborn as sns


sns.heatmap(df1.corr(), annot=True)
plt.show()


def subplots(df1):
    _,axs = plt.subplots(2,2,figsize=(10,6))
    sns.scatterplot(x='Tyro_300',y='Cheeky',ax=axs[0,0],data=df1)
    sns.scatterplot(x='0.8',y='4.6',ax=axs[0,1],data=df1)
    sns.scatterplot(x='129',y='0.8',ax=axs[1,0],data=df1)
    sns.barplot(x='China',y='4.6',ax=axs[1,1],data=df1)
    plt.tight_layout()
    plt.show()
    
    
subplots(df1)



print(df.dtypes)

df['Tyro_300'].value_counts()


X = df.drop(['129','Tyro_300'],axis=1)
y = df['129']

df['Tyro_300'].describe()

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)


from sklearn.preprocessing import OneHotEncoder,PolynomialFeatures
ohe = OneHotEncoder(sparse_output=False)
poly = PolynomialFeatures(degree=2)

from sklearn.compose import make_column_transformer

ct = make_column_transformer(
    (ohe,X.select_dtypes(include='object').columns),
    (poly,X.select_dtypes(include=['float64','int64']).columns),remainder='passthrough')


X_train_transformed = ct.fit_transform(X_train)
X_test_transformed = ct.transform(X_test)

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LinearRegression,Lasso,Ridge

lr = LinearRegression()
ridge = Ridge()
lasso = Lasso()

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,BaggingRegressor

RFR = RandomForestRegressor()
GBR = GradientBoostingRegressor()
BR = BaggingRegressor()


from sklearn.metrics import mean_squared_error,r2_score


def evaluate_model(X_train,X_test,y_train,y_test,model):
    pipe = make_pipeline(ct,model).fit(X_train,y_train)
    pred = pipe.predict(X_test)
    r2 = r2_score(y_test, pred)
    mse = mean_squared_error(y_test,pred)
    print(f'{model.__class__.__name__}, --R2-- {r2*100:.2f}; --MSE-- {mse:.2f}')
    return pred


lr_pred = evaluate_model(X_train, X_test, y_train, y_test, lr)
ridge_pred = evaluate_model(X_train, X_test, y_train, y_test, ridge)
lasso_pred = evaluate_model(X_train, X_test, y_train, y_test, lasso)
RFR_pred = evaluate_model(X_train, X_test, y_train, y_test, RFR)
GBR_pred = evaluate_model(X_train, X_test, y_train, y_test, GBR)
BR_pred = evaluate_model(X_train, X_test, y_train, y_test, BR)














