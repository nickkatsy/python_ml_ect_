import pandas as pd
import warnings
warnings.filterwarnings('ignore')



df = pd.read_csv('C:/ML/python/data/PRODUCT_SALES.csv',delimiter=',')

df.info()
df.isna().sum()
df.nunique()
df['Customer_Gender'] = df['Customer_Gender'].map({"M":0,"F":1})

df['Age_Group'].value_counts()


age_groups = ['Adults (35-64)','Young Adults (25-34)','Youth (<25)',"Seniors (64+)"]

df['Age_Group'] = pd.Categorical(df.Age_Group,categories=age_groups,ordered=True)
df['Adults'] = pd.get_dummies(df['Age_Group'],prefix='Age_Group').iloc[:,0:1]
df['Young_Adults'] = pd.get_dummies(df.Age_Group,prefix='Age_Group').iloc[:,1:2]
df['Youth'] = pd.get_dummies(df.Age_Group,prefix='Age_Group').iloc[:,2:3]



df.drop_duplicates(inplace=True)

df.isna().sum()

df['Product_Category'].value_counts()



df['Country'].value_counts()

df['Month'].nunique()

df.isna().sum()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()


df1 = df.copy()


for i in df1:
    df1[i] = le.fit_transform(df1[i])






import seaborn as sns
import matplotlib.pyplot as plt


def product_profit(df):
    sns.barplot(x=df['Product_Category'],y=df['Profit'])
    plt.xlabel('Name of Product')
    plt.ylabel('Profit')
    plt.show()



product_profit(df)


def customer_age(df):
    sns.kdeplot(x='Customer_Age',data=df)
    plt.title('Age of Customer')
    plt.tight_layout()
    plt.show()
    
customer_age(df)
    
    

def order_quant_type(df):
    sns.barplot(x='Product_Category',y='Order_Quantity',data=df)
    plt.xlabel('Name of Product')
    plt.ylabel('Number of Units Ordered')
    plt.show()


order_quant_type(df)


def bar_age(df):
    sns.barplot(x=df['Age_Group'],y=df['Revenue'])
    plt.ylabel('Revenue')
    plt.title('Revenue by Age')
    plt.show()


bar_age(df)


def countplot_customer_gender(dataframe):
    sns.countplot(x='Customer_Gender',data=df)
    plt.show()


countplot_customer_gender(df)

df['profit'] = df['Revenue'] - df['Cost']

from scipy.optimize import minimize
import sympy as sym
import numpy as np

df['MR'] = sym.diff(df['Revenue'])
df['MC'] = sym.diff(df['Cost'])

df['pi'] = df['MR'] - df['MC']

df['max_profit'] = np.max(np.where(df['MR'] == df['MC']))
print(df['max_profit'])

def objective(x):
    MR = x[0]
    MC = x[1]
    lambda x: MR == MC
    return (np.linalg.norm(x))

min_cost = np.min(df['Cost'])
max_cost = np.max(df['Cost'])
min_revenue= np.min(df['Revenue'])
max_revenue = np.max(df['Revenue'])

b1 = [min_revenue,max_revenue]
b2 = [min_cost,max_cost]
b = (b1,b2)
x0 = [0.001,10000]
result = minimize(objective, x0,bounds=b,method='SLSQP')
print(result)










def subplots(df):
    _,axs = plt.subplots(2,3,figsize=(15,7))
    sns.barplot(x='Revenue',y='Country',ax=axs[0,0],data=df)
    axs[0,0].set_title('Profit by Country')
    
    sns.scatterplot(x='Order_Quantity',y='Customer_Age',ax=axs[0,1],data=df)
    axs[0,1].set_title('Order_Quantity Based on Customers Age')
    
    
    sns.barplot(x='Product_Category',y='Profit',ax=axs[0,2],data=df)
    axs[1,0].set_title('Type of Product and Profit')
    
    
    sns.scatterplot(x='Revenue',y='Cost',ax=axs[1,0],data=df1)
    sns.lineplot(x='MR',y='MC',ax=axs[0,1],data=df,palette='pastel')
    axs[1,1].set_title('Revenue Vs Cost')
    
    sns.violinplot(x='Customer_Age',y='Country',ax=axs[1,1],data=df)
    axs[1,1].set_title('Customer Age By Country')
    
    sns.boxplot(x='Customer_Age',y='Product_Category',ax=axs[1,2],data=df)
    axs[1,2].set_title('Customer Age Vs Product Category')


plt.show()



subplots(df)






df = df.drop(['Month','Day','Year','Date','Age_Group'],axis=1)

X = df.drop('Revenue',axis=1)
y = df['Revenue']


df.isna().sum()



from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)


from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False)
from sklearn.impute import SimpleImputer

si = SimpleImputer(strategy='most_frequent')

from sklearn.compose import make_column_transformer

ct = make_column_transformer(
    (ohe,X.select_dtypes(include='object').columns),
    (si,X.select_dtypes(include=['int64','float64']).columns),remainder='passthrough')



ct.fit_transform(X)


from sklearn.pipeline import make_pipeline


from sklearn.linear_model import LinearRegression,Lasso,Ridge
lr = LinearRegression()
ridge = Ridge()
lasso = Lasso()
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor
rfr = RandomForestRegressor()
BR = BaggingRegressor()





from sklearn.metrics import r2_score,mean_squared_error

def evaluate_model(X_train,X_test,y_train,y_test,model):
    pipe = make_pipeline(ct,model).fit(X_train,y_train)
    pred = pipe.predict(X_test)
    r2 = r2_score(y_test, pred)
    mse = mean_squared_error(y_test,pred)
    print(f'{model.__class__.__name__}, --r2-- {r2*100:.2f}; --MSE-- {mse:.2f}')
    return pred


lr_pred = evaluate_model(X_train, X_test, y_train, y_test, lr)
rfr_pred = evaluate_model(X_train, X_test, y_train, y_test, rfr)
BR_pred = evaluate_model(X_train, X_test, y_train, y_test, BR)
ridge_pred = evaluate_model(X_train, X_test, y_train, y_test,ridge)
lasso_pred = evaluate_model(X_train, X_test, y_train, y_test, lasso)



# binomial now
X_youth = df.drop('Youth',axis=1)
y_youth = df['Youth']

X_youth_train,X_youth_test,y_youth_train,y_youth_test = train_test_split(X_youth,y_youth,test_size=.20,random_state=42)



ct = make_column_transformer(
    (ohe,X_youth.select_dtypes(include='object').columns),
    (si,X_youth.select_dtypes(include=['float64','int64']).columns),remainder='passthrough')


ct.fit_transform(X_youth)


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
rfc = RandomForestClassifier()
BC = BaggingClassifier()

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()


from sklearn.metrics import roc_auc_score,accuracy_score,confusion_matrix

def evalL(X_youth_train,X_youth_test,y_youth_train,y_youth_test,model):
    pipe = make_pipeline(ct,model).fit(X_youth_train,y_youth_train)
    pred = pipe.predict(X_youth_test)
    pred_prob = pipe.predict_proba(X_youth_test)[:,1]
    acc = accuracy_score(y_youth_test,pred)
    con = confusion_matrix(y_youth_test, pred)
    roc = roc_auc_score(y_youth_test,pred_prob)
    print(f'{model.__class__.__name__}, --ACC-- {acc*100:.2f}%; --ROC-- {roc*100:.2f}%')
    print('confusion matrix',con)
    return pred,pred_prob


clf_pred,clf_pred_prob = evalL(X_youth_train, X_youth_test, y_youth_train, y_youth_test,clf)
rfc_pred,rfc_pred_prob = evalL(X_youth_train, X_youth_test, y_youth_train, y_youth_test,rfc)
BC_pred,BC_pred_prob = evalL(X_youth_train, X_youth_test, y_youth_train, y_youth_test, BC)
nb_pred,nb_pred_prob = evalL(X_youth_train, X_youth_test, y_youth_train, y_youth_test, nb)







