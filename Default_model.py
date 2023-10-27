import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('https://raw.githubusercontent.com/LeeMorinUCF/QMB6358F23/main/demo_14_linear_models_in_python/credit_data.csv')

df.info()
df.isna().sum()
df.isnull().sum()
df.nunique()
df.corr()

import matplotlib.pyplot as plt
import seaborn as sns

# Visualize the correlation matrix
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

from sklearn.model_selection import train_test_split

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42)

from sklearn.linear_model import LogisticRegression

# Logistic Regression Model
clf = LogisticRegression().fit(X_train, y_train)
clf_pred = clf.predict(X_test)
clf_pred_prob = clf.predict_proba(X_test)[::,1]

from sklearn.ensemble import RandomForestClassifier

# Random Forest Model
rfc = RandomForestClassifier().fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
rfc_pred_prob = rfc.predict_proba(X_test)[::,1]

from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

# Model Performance
acc_logistic_regression = accuracy_score(y_test, clf_pred)
roc_clf = roc_auc_score(y_test, clf_pred_prob)
acc_rfc = accuracy_score(y_test, rfc_pred)
roc_rfc = roc_auc_score(y_test, clf_pred_prob)


# results
print('Logistic Regression Accuracy:', acc_logistic_regression * 100)
print('Logistic Regression ROC AUC:', roc_clf * 100)
print('Random Forest Accuracy:', roc_rfc * 100)
print('Random Forest ROC AUC:', roc_rfc * 100)

fpr_clf, tpr_clf, _ = roc_curve(y_test, clf_pred_prob)
fpr_rfc, tpr_rfc, _ = roc_curve(y_test, rfc_pred_prob)

# ROC for RFC and Logistic Regression and Random Forest
plt.figure(figsize=(8, 6))
plt.plot(fpr_clf, tpr_clf, label='Logistic Regression')
plt.plot(fpr_rfc, tpr_rfc, label='Random Forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()
