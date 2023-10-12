import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('https://raw.githubusercontent.com/nickkatsy/python_ml_ect_/master/credit_risk.csv',delimiter=',')

df.info()
df.isna().sum()

df.nunique()
df.Default.value_counts()


df['Default'] = pd.get_dummies(df.Default,prefix='Default').iloc[:,0:1]


import seaborn as sns
import matplotlib.pyplot as plt


sns.heatmap(df.corr(),annot=True)

def descriptive_subplots(dataframe):
    plt_,axs = plt.subplots(3,3,figsize=(10,6))
    sns.countplot(df,x='Home',ax=axs[0,0])
    sns.countplot(df,x='Status',ax=axs[0,1])
    sns.scatterplot(df,x='Age',y='Income',ax=axs[0,2])
    sns.violinplot(df[['Status']],ax=axs[1,0])
    sns.countplot(df,x='Default',ax=axs[1,1])
    sns.kdeplot(df,x='Age',ax=axs[1,2])
    sns.countplot(df,x='Intent',ax=axs[2,0])
    sns.scatterplot(df[['Income']],ax=axs[2,1])
    sns.countplot(df,x='Status',ax=axs[2,2])
    plt.show()

descriptive_subplots(df)

df = df.drop('Id',axis=1)


X = df.drop('Default',axis=1)
y = df[['Default']]

from sklearn.model_selection import train_test_split


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)



from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()


from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer


ohe = OneHotEncoder()
imp = SimpleImputer()


ct = make_column_transformer(
    (ohe,['Home','Intent']),(imp,['Emp_length','Rate']),remainder='passthrough')


ct.fit_transform(X)

from sklearn.pipeline import make_pipeline

# Pipeline for logistic regression
clf_pipe = make_pipeline(ct,clf).fit(X_train,y_train)
clf_pred = clf_pipe.predict(X_test)
clf_pred_prob = clf_pipe.predict_proba(X_test)[::,1]

# Making a pipeline for Random Forrest Classifier
rfc_pipe = make_pipeline(ct,rfc).fit(X_train,y_train)
rfc_pred = rfc_pipe.predict(X_test)
rfc_pred_prob = rfc_pipe.predict_proba(X_test)[::,1]

# Results
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve

acc_clf = accuracy_score(y_test, clf_pred)
print('accuracy of logistic regression model',acc_clf*100)

clf_roc = roc_auc_score(y_test, clf_pred_prob)
print('roc logistic regression= ',clf_roc*100)

acc_rfc = accuracy_score(y_test, rfc_pred)
print('Random Forest Accuracy= ',acc_rfc*100)

roc_rfc = roc_auc_score(y_test, rfc_pred_prob)
print('roc for Random Forest= ',roc_rfc*100)

# plot of ROC/AUC from best performing model(rfc)
fpr, tpr, _ = roc_curve(y_test,  rfc_pred_prob)

plt.plot(fpr,tpr)
plt.title('Random Forest')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

