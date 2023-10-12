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

def subplot_comparisons(dataframe):
    plt__,axs = plt.subplots(2,3,figsize=(12,6))
    sns.violinplot(df,x='A',ax=axs[0,0])
    sns.countplot(df,x='B',ax=axs[0,1])
    sns.countplot(df,x='C',ax=axs[1,0])
    sns.countplot(df,x='D',ax=axs[1,1])
    sns.lineplot(df,x='amount',y='default',ax=axs[0,2])
    sns.scatterplot(df,x='bmaxrate',y='amount',ax=axs[1,2])
    plt.show()


subplot_comparisons(df)

##  Subplots of Risks of Default based on rating



def rating_subplots(dataframe):
    plt_default,axs2 = plt.subplots(2,2,figsize=(12,6))
    sns.barplot(df,x='AA',y='default',ax=axs2[0,0])
    sns.barplot(df,x='B',y='default',ax=axs2[0,1])
    sns.barplot(df,x='C',y='default',ax=axs2[1,0])
    sns.lineplot(df,x='D',y='default',ax=axs2[1,1])
    plt.show()

rating_subplots(df)

### Risk of defaulting based on all variables

X = df.drop('default',axis=1)
y = df[['default']]


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)



# Logistic Regression
from sklearn.linear_model import LogisticRegression


clf = LogisticRegression().fit(X_train,y_train)

clf_pred = clf.predict(X_test)


clf_pred_prob = clf.predict_proba(X_test)[::,1]


#Random Forrest

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier().fit(X_train,y_train)

rfc_pred = rfc.predict(X_test)

rfc_pred_prob = rfc.predict_proba(X_test)[::,1]


# Default Results

from sklearn.metrics import accuracy_score, roc_auc_score


#logistic regression

acc_logistic_regression = accuracy_score(y_test, clf_pred)
print('accuaracy using logistic regression model',acc_logistic_regression*100)

roc_clf = roc_auc_score(y_test, clf_pred_prob)
print('roc score using logistic regression',roc_clf*100)


## Random Forrest Results for default risk


accuracy_random_forrest = accuracy_score(y_test, rfc_pred)
print('accuracy Random Forest Classification',accuracy_random_forrest*100)

roc_rfc = roc_auc_score(y_test, rfc_pred_prob)
print('roc for defaulting using random forest',roc_rfc*100)
