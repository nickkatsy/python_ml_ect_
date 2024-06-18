import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("C:/ML/python/data/WA_Fn-UseC_-Telco-Customer-Churn.csv",delimiter=',')

df.info()
df.isna().sum()
df.duplicated().sum()
print(df.dtypes)
print(df.nunique())
df.drop('customerID',axis=1,inplace=True)
df['TotalCharges'] = df['TotalCharges'].str.replace(" ","")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])



df['gender'].value_counts().plot(kind='pie',autopct = '%1.1f%%')


df['Contract'].value_counts().plot(kind='bar',rot=0)


df['Dependents'].value_counts().plot(kind='pie',autopct="%1.1f%%")


df['InternetService'].value_counts().plot(kind='bar',rot=0)


sns.histplot(df['DeviceProtection'])
plt.show()


obj_cols = df.select_dtypes(include='object')

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for i in obj_cols:
    obj_cols[i] = le.fit_transform(obj_cols[i])


num_cols = df.select_dtypes(include=['float64','int32'])


def desc_stats(obj_cols):
    f,axs = plt.subplots(2,2,figsize=(10,6))
    sns.barplot(x='gender',y='Churn',data=obj_cols,ax=axs[0,0])
    sns.boxplot(x='Dependents',y='gender',data=obj_cols,ax=axs[0,1])
    sns.barplot(x='PaymentMethod',y='Churn',data=obj_cols,ax=axs[1,0])
    sns.barplot(x='InternetService',y='Dependents',data=obj_cols,ax=axs[1,1])
    plt.show()
    
desc_stats(obj_cols)





df.isnull().sum()

df.dropna(inplace=True)
df.duplicated().sum()
df.drop_duplicates(inplace=True)

df['Churn'] = [1 if X == 'Yes' else 0 for X in df['Churn']]

X = df.drop('Churn',axis=1)
y = df['Churn']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)



from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
ms = MinMaxScaler()


from sklearn.compose import make_column_transformer
ct = make_column_transformer(
    (ohe,X.select_dtypes(include='object').columns),
    (ms,X.select_dtypes(include=['float64','int32']).columns),remainder='passthrough')


ct.fit_transform(X)

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,BaggingClassifier

RFC = RandomForestClassifier()
GBC = GradientBoostingClassifier()
BC = BaggingClassifier()

from xgboost import XGBClassifier
xgb = XGBClassifier()

from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score,f1_score,confusion_matrix


def evaluate_(X_train,X_test,y_train,y_test,model):
    pipe = make_pipeline(ct,model).fit(X_train,y_train)
    pred = pipe.predict(X_test)
    pred_prob=  pipe.predict_proba(X_test)[:,1]
    acc = accuracy_score(y_test,pred)
    roc = roc_auc_score(y_test,pred_prob)
    f1 = f1_score(y_test, pred)
    print('f1 score: ',f1*100)
    print(f'{model.__class__.__name__}, --ACC-- {acc*100:.2f}%; --ROC-- {roc*100:.2f}%')
    return pred,pred_prob


lr_pred,lr_pred_prob = evaluate_(X_train, X_test, y_train, y_test,lr)
RFC_pred,RFC_pred_prob = evaluate_(X_train, X_test, y_train, y_test, RFC)
GBC_pred,GBC_pred_prob=  evaluate_(X_train, X_test, y_train, y_test, GBC)
BC_pred,BC_pred_prob = evaluate_(X_train, X_test, y_train, y_test, BC)
xgb_pred,xgb_pred_prob = evaluate_(X_train, X_test, y_train, y_test,xgb)

def confusion_matrix_plot(X_train,X_test,y_train,y_test,model):
    pipe = make_pipeline(ct,model).fit(X_train,y_train)
    pred = pipe.predict(X_test)
    con = confusion_matrix(y_test,pred)
    heatmap = sns.heatmap(con,annot=True,fmt="d",cmap="coolwarm")
    heatmap.set_title(f'Confusion Matrix for {model.__class__.__name__}')
    return heatmap

lr_confusion_matrix = confusion_matrix_plot(X_train, X_test, y_train, y_test,lr)
RFC_confusion_matrix=  confusion_matrix_plot(X_train, X_test, y_train, y_test, RFC)
GBC_confusion_matrix = confusion_matrix_plot(X_train, X_test, y_train, y_test, GBC)
BC_confusion_matrix = confusion_matrix_plot(X_train, X_test, y_train, y_test,BC)
xgb_confusion_matrix = confusion_matrix_plot(X_train, X_test, y_train, y_test, xgb)


def ROC(y_test,y_pred_prob,model):
    fpr,tpr, _ = roc_curve(y_test,y_pred_prob)
    plt.plot(fpr,tpr,label=model.__class__.__name__)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Chosen Models")
    


ROC(y_test,lr_pred_prob,lr)
ROC(y_test,RFC_pred_prob,RFC)
ROC(y_test,GBC_pred_prob,GBC)
ROC(y_test,BC_pred_prob,BC)
ROC(y_test,xgb_pred_prob,xgb)
plt.legend()
plt.plot()


from sklearn.model_selection import cross_val_score

def cross_val_(X_train,X_test,y_train,y_test,model):
    pipe = make_pipeline(ct,model).fit(X,y)
    cv_scores = cross_val_score(pipe, X,y,cv=10,scoring='roc_auc').max()
    print(f'{model.__class__.__name__}, --Cross validation scores 10 fold-- {cv_scores*100:.2f}%')
    return cv_scores

lr_scores = cross_val_(X_train, X_test, y_train, y_test, lr)
RFC_scores = cross_val_(X_train, X_test, y_train, y_test, RFC)
GBC_scores = cross_val_(X_train, X_test, y_train, y_test, GBC)
BC_scores = cross_val_(X_train, X_test, y_train, y_test, BC)
xgb_scores = cross_val_(X_train, X_test, y_train, y_test,xgb)







