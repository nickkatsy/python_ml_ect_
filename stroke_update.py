import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('https://raw.githubusercontent.com/nickkas/python_ml_ect_/master/stroke.csv')
df.info()

df.isna().sum()
df.nunique()

df['gender'].value_counts()
df['gender'] = pd.get_dummies(df.gender,prefix='gender').iloc[:,0:1]

df['id'].value_counts()

df['stroke'].value_counts()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df = df.drop('id',axis=1)
df1 = df.copy()

for i in df1:
    df1[i] = le.fit_transform(df1[i])
    

import seaborn as sns


sns.heatmap(df1.corr(), annot=True)
plt.show()


def desc(df1):
    _,axs = plt.subplots(2,2,figsize=(10,6))
    sns.barplot(x='gender',y='stroke',ax=axs[0,0],data=df1)
    sns.lineplot(x='age',y='smoking_status',ax=axs[0,1],data=df1)
    sns.lineplot(x='bmi',y='age',ax=axs[1,0],data=df1)
    sns.barplot(x='heart_disease',y='stroke',ax=axs[1,1],data=df1)
    plt.show()
    plt.tight_layout()



desc(df1)








X = df.drop('stroke',axis=1)
y = df[['stroke']]


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB






from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer

ohe = OneHotEncoder()
imputer = SimpleImputer(strategy='mean')


ct = make_column_transformer(
    (ohe, X.select_dtypes(include='object').columns),
    (imputer, X.select_dtypes(['int64', 'float64']).columns),remainder='passthrough')




ct.fit_transform(X)

from sklearn.pipeline import make_pipeline

rfc = RandomForestClassifier()

nb = GaussianNB()
lr = LogisticRegression()

knn = KNeighborsClassifier(n_neighbors=7)

GBC = GradientBoostingClassifier()
BC = BaggingClassifier()


from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score,f1_score,confusion_matrix



def evaluate_model(X_train,X_test,y_train,y_test,model):
    pipe = make_pipeline(ct,model).fit(X_train,y_train)
    pred = pipe.predict(X_test)
    pred_prob = pipe.predict_proba(X_test)[:,1]
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    con = confusion_matrix(y_test, pred)
    roc = roc_auc_score(y_test,pred_prob)
    print('confusion matrix: ',con)
    print(f'{model.__class__.__name__}, --Accuracy Score-- {acc*100:.2f}%; --ROC-- {roc*100:.2f}%; --F1-- {f1*100:.2f}%')
    return pred,pred_prob





lr_pred,lr_pred_prob = evaluate_model(X_train, X_test, y_train, y_test, lr)
rfc_pred,rfc_pred_prob = evaluate_model(X_train, X_test, y_train, y_test, rfc)
nb_pred,nb_pred_prob = evaluate_model(X_train, X_test, y_train, y_test, nb)
BC_pred,BC_pred_prob = evaluate_model(X_train, X_test, y_train, y_test,BC)
gbc_pred,gbc_pred_prob = evaluate_model(X_train, X_test, y_train, y_test,GBC)
knn_pred,knn_pred_prob = evaluate_model(X_train, X_test, y_train, y_test,knn)



def ROC_Curve(y_test,y_pred_prob,model):
    fpr,tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.plot(fpr,tpr,label=model.__class__.__name__)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')



ROC_Curve(y_test,lr_pred_prob,lr)
ROC_Curve(y_test,rfc_pred_prob,rfc)
ROC_Curve(y_test,gbc_pred_prob,GBC)
ROC_Curve(y_test,nb_pred_prob,nb)
ROC_Curve(y_test, knn_pred_prob,knn)
ROC_Curve(y_test,BC_pred_prob,BC)
plt.legend()
plt.show()
