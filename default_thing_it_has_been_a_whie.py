import pandas as pd
import warnings
warnings.filterwarnings('ignore')



df = pd.read_csv('https://raw.githubusercontent.com/LeeMorinUCF/QMB6358F23/main/demo_14_linear_models_in_python/credit_data.csv')

df.info()

df.isna().sum()


df.isnull().sum()


df.nunique()


df.corr()


import seaborn as sns
import matplotlib.pyplot as plt


sns.heatmap(df.corr(), annot=True)


plt__,axs = plt.subplots(2,3,figsize=(12,6))
plt_1 = sns.violinplot(df,x='A',ax=axs[0,0])
plt_2 = sns.countplot(df,x='B',ax=axs[0,1])
plt__3 = sns.countplot(df,x='C',ax=axs[1,0])
plt___4 = sns.countplot(df,x='D',ax=axs[1,1])
plt____5 = sns.lineplot(df,x='amount',y='default',ax=axs[0,2])
plt____6 = sns.scatterplot(df,x='bmaxrate',y='amount',ax=axs[1,2])




##  Risk of Default

plt_default,axs2 = plt.subplots(2,2,figsize=(12,6))
plt__1 = sns.barplot(df,x='AA',y='default',ax=axs2[0,0])
plt__2 = sns.barplot(df,x='B',y='default',ax=axs2[0,1])
plt__3 = sns.barplot(df,x='C',y='default',ax=axs2[1,0])
plt_4 = sns.lineplot(df,x='D',y='default',ax=axs2[1,1])



### Risk of defaulting based on all variables


X = df.drop(['A','B','C','D','default'],axis=1)
y = df[['default']]


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)


from sklearn.linear_model import LogisticRegression


clf = LogisticRegression().fit(X_train,y_train)

clf_pred = clf.predict(X_test)


clf_pred_prob = clf.predict_proba(X_test)[::,1]


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier().fit(X_train,y_train)

rfc_pred = rfc.predict(X_test)

rfc_pred_prob = rfc.predict_proba(X_test)[::,1]


# Default Results

from sklearn.metrics import accuracy_score, roc_auc_score


## Logistic Regression


accuracy_clf = accuracy_score(y_test, clf_pred)
print('The accuracy of the logistic regression model for the risk of default',accuracy_clf*100)

roc_clf = roc_auc_score(y_test, clf_pred_prob)
print('ROC Logistic Regression',roc_clf*100)


## Random Forrest Results for default risk


accuracy_random_forrest = accuracy_score(y_test, clf_pred)
print('accuracy Random Forest Classification',accuracy_random_forrest*100)

roc_rfc = roc_auc_score(y_test, rfc_pred_prob)
print('roc for defaulting using random forest',roc_rfc*100)
















