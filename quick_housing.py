import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('https://raw.githubusercontent.com/LeeMorinUCF/QMB6358F23/main/demo_14_linear_models_in_python/housing_data.csv')


df.info()

df.isna().sum()

df.corr()

df.nunique()


df = df.drop('obsn_num',axis=1)


import seaborn as sns
import matplotlib.pyplot as plt

def basic_subplots(df):
  plt____,axs = plt.subplots(2,2,figsize=(12,6))
  sns.scatterplot(df,x='income',y='house_price',ax=axs[0,0])
  sns.kdeplot(df,x='in_cali',ax=axs[0,1])
  sns.violinplot(df['income'],ax=axs[1,0])
  sns.distplot(df['earthquake'],ax=axs[1,1])
  plt.show()


basic_subplots(df)

# Selection

X = df.drop('in_cali',axis=1)
y = df[['in_cali']]


from sklearn.model_selection import train_test_split


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)


# scaling the features

from sklearn.preprocessing import StandardScaler


sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# Creating Classifier
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression().fit(X_train_scaled,y_train)

clf_pred = clf.predict(X_test_scaled)

clf_pred_prob = clf.predict_proba(X_test_scaled)[::,1]


from sklearn.naive_bayes import GaussianNB

nb = GaussianNB().fit(X_train_scaled,y_train)
nb_pred = nb.predict(X_test_scaled)
nb_pred_prob = nb.predict_proba(X_test_scaled)[::,1]

from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier().fit(X_train_scaled,y_train)
gbc_pred = gbc.predict(X_test_scaled)
gbc_pred_prob = gbc.predict_proba(X_test_scaled)[::,1]

# scores
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve

acc_clf = accuracy_score(y_test, clf_pred)
print(f'the accuaracy of the logistic regression model: {acc_clf}')

roc_clf = roc_auc_score(y_test,clf_pred_prob)
print(f'the roc_auc score of the logistic regression model: {roc_clf}')


acc_nb = accuracy_score(y_test, nb_pred)
print(f'the accuracy using Naive Bayes: {acc_nb}')
roc_nb = roc_auc_score(y_test, nb_pred_prob)
print(roc_nb)

acc_gbc = accuracy_score(y_test, gbc_pred)
print(f'the accuracy using Gradient Boost: {acc_gbc}')

roc_gbc = roc_auc_score(y_test, gbc_pred_prob)
print(f'the roc_auc score using Gradient Boost: {roc_gbc}')



# cross-validation results
from sklearn.model_selection import cross_val_score

cv_clf = cross_val_score(clf,X_train_scaled,y_train,cv=10,scoring='roc_auc').mean()
print(f'the roc_auc score using 10-fold cv for logistic regression model: {cv_clf}')

cv_nb = cross_val_score(nb,X_train_scaled, y_train,cv=10,scoring='roc_auc').mean()
print(f'the 10-fold cross validation score using the Naive Bayes Classifier: {cv_nb}')

cv_gbc = cross_val_score(gbc, X_train_scaled,y_train,cv=10,scoring='roc_auc').mean()
print(f'the score of the Gradient Boost classifier using 10-fold cross-validation: {cv_gbc}')


def roc_curve_plot(y_test, y_pred_prob, model_name):
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.plot(fpr,tpr,label=model_name)
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

roc_curve_plot(y_test,clf_pred_prob,'Logistic Regression')
roc_curve_plot(y_test,nb_pred_prob,'Naive Bayes')
roc_curve_plot(y_test,gbc_pred_prob,'Gradient Boosting')
plt.legend()
plt.show()

