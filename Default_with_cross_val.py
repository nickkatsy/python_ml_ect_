import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('https://raw.githubusercontent.com/LeeMorinUCF/QMB6358F23/main/demo_14_linear_models_in_python/credit_data.csv')

df.info()
df.isna().sum()
df.isnull().sum()
df.nunique()
df.corr()


df['default'] = 1 - df['default']

import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True)
plt.title('Correlation Matrix')
plt.show()

def subplot_comparisons(dataframe):
    _, axs = plt.subplots(2, 3, figsize=(12, 6))
    sns.violinplot(df, x='A', ax=axs[0,0])
    sns.countplot(df, x='B', ax=axs[0,1])
    sns.countplot(df, x='C', ax=axs[1,0])
    sns.countplot(df, x='D', ax=axs[1,1])
    sns.lineplot(df, x='amount', y='default', ax=axs[0,2])
    sns.scatterplot(df, x='bmaxrate', y='amount', ax=axs[1,2])
    plt.show()

subplot_comparisons(df)

def rating_subplots(dataframe):
    plt__, axs = plt.subplots(2, 2,figsize=(12, 6))
    sns.barplot(df, x='AA', y='default', ax=axs[0,0])
    sns.barplot(df, x='B', y='default', ax=axs[0,1])
    sns.barplot(df, x='C', y='default', ax=axs[1,0])
    sns.lineplot(df, x='D', y='default', ax=axs[1,1])
    plt.show()

rating_subplots(df)

X = df.drop('default', axis=1)
y = df['default']


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit_transform(X)

from sklearn.linear_model import LogisticRegression


clf = LogisticRegression().fit(X,y)
clf_pred = clf.predict(X)


from sklearn.ensemble import RandomForestClassifier

# Random Forest Model
rfc = RandomForestClassifier().fit(X,y)
rfc_pred = rfc.predict(X)


from sklearn.metrics import accuracy_score

# Model Performance
acc_logistic_regression = accuracy_score(y, clf_pred)
print(f' the accuracy using Logistic Regression: {acc_logistic_regression*100}')


acc_rfc = accuracy_score(y, rfc_pred)
print(f'the accuracy of random forest for this model: {acc_rfc*100}')


from sklearn.model_selection import cross_val_predict
clf_pred_prob_cv = cross_val_predict(clf, X, y, cv=5, method='predict_proba').max()
rfc_pred_prob_cv = cross_val_predict(rfc, X, y, cv=5, method='predict_proba').max()


print(f'ROC using 5-fold cross-validation with Logistic Regression: {clf_pred_prob_cv*100}')
print(f'ROC using 5-fold cross-validation with Random Forest: {rfc_pred_prob_cv*100}')
