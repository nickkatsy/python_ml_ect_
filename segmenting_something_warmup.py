import pandas as pd
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('https://raw.githubusercontent.com/nickkatsy/python_ml_ect_/master/shopping_trends.csv')


df.info()
df.columns = df.columns.str.replace(' ','_')
df.nunique()
df.isna().sum()
print(df.dtypes)
df['Gender'] = df['Gender'].map({'Male':0,'Female':1})
df['Category'].value_counts()
df['Shipping_Type'].value_counts()

df = df.drop('Customer_ID',axis=1)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df1 = df.copy()


for i in df1:
    df1[i] = le.fit_transform(df1[i])

import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(df1.corr(), annot=True)
plt.show()


def desc(df1):
    _,axs = plt.subplots(2,2,figsize=(10,6))
    sns.barplot(x='Category',y='Gender',ax=axs[0,0],data=df1)
    sns.lineplot(x='Age',y='Item_Purchased',ax=axs[0,1],data=df1)
    sns.lineplot(x='Purchase_Amount_(USD)',y='Age',ax=axs[1,0],data=df1)
    sns.barplot(x='Season',y='Shipping_Type',ax=axs[1,1],data=df1)
    plt.show()


desc(df1)

df['Subscription_Status'] = [1 if X == 'Yes' else 0 for X in df['Subscription_Status']]

X = df.drop('Subscription_Status',axis=1)
y = df['Subscription_Status']


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)

X.dtypes
from sklearn.preprocessing import OneHotEncoder,StandardScaler


ohe = OneHotEncoder(sparse_output=False)
sc = StandardScaler()

from sklearn.compose import make_column_transformer

ct = make_column_transformer(
    (ohe,X.select_dtypes(include='object').columns),
    (sc,X.select_dtypes(include=['float64','int64']).columns),remainder='passthrough')


ct.fit_transform(X)

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,GradientBoostingClassifier


GBC = GradientBoostingClassifier()
RFC = RandomForestClassifier()
BC = BaggingClassifier()

from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score


def evaluate_model(X_train,X_test,y_train,y_test,model):
    pipe = make_pipeline(ct,model).fit(X_train,y_train)
    pred = pipe.predict(X_test)
    pred_prob = pipe.predict_proba(X_test)[:,1]
    acc = accuracy_score(y_test,pred)
    roc = roc_auc_score(y_test,pred_prob)
    print(f'{model.__class__.__name__}, --ACC-- {acc*100:.2f}%; --ROC-- {roc*100:.2f}%')
    return pred,pred_prob



lr_pred,lr_pred_prob = evaluate_model(X_train, X_test, y_train, y_test, lr)
GBC_pred,GBC_pred_prob = evaluate_model(X_train, X_test, y_train, y_test, GBC)
RFC_pred,RFC_pred_prob = evaluate_model(X_train, X_test, y_train, y_test, RFC)
BC_pred,BC_pred_prob = evaluate_model(X_train, X_test, y_train, y_test, BC)


def ROC(y_test,y_pred_prob,model):
    fpr,tpr,_ = roc_curve(y_test, y_pred_prob)
    plt.plot(fpr,tpr,label=model.__class__.__name__)
    plt.xlabel('False Positive')
    plt.ylabel('True Positive Rate')

ROC(y_test,lr_pred_prob,lr)
ROC(y_test,GBC_pred_prob,GBC)
ROC(y_test,RFC_pred_prob,RFC)
plt.legend()
plt.show()











