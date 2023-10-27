import pandas as pd
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('C:/ml/python/data/creditcard.csv', delimiter=',')


print(df.info())


print(df.isna().sum())
print(df.isnull().sum())


print(df.nunique())


import matplotlib.pyplot as plt
import seaborn as sns


#basic plots

def visualize_data(df):
    plt_, axs = plt.subplots(2, 2, figsize=(12, 6))
    sns.barplot(data=df, x='Class', y='Amount', ax=axs[0, 0])
    sns.distplot(df['V1'], ax=axs[0,1])
    sns.kdeplot(data=df, x='V2', ax=axs[1, 0])
    sns.kdeplot(data=df, x='Class', ax=axs[1, 1])
    plt.show()


visualize_data(df)


X = df.drop('Class', axis=1)
y = df['Class']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from imblearn.over_sampling import SMOTE

smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

#Logistic Regression
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression().fit(X_train_resampled, y_train_resampled)
clf_pred = clf.predict(X_test)
clf_pred_prob = clf.predict_proba(X_test)[::,1]


from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score

roc_auc = roc_auc_score(y_test, clf_pred_prob)
accuracy = accuracy_score(y_test, clf_pred)

print(f'Logistic Regression - ROC AUC: {roc_auc:.2f}, Accuracy: {accuracy:.2f}')


# Plotting the ROC curve
fpr, tpr, _ = roc_curve(y_test, clf_pred_prob)
plt.plot(fpr, tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.show()

# 5-fold Cross Validation

from sklearn.model_selection import cross_validate

cv_results = cross_validate(clf, X_train_resampled, y_train_resampled, cv=5, scoring='roc_auc', return_train_score=False)
print('Mean ROC AUC:', cv_results['test_score'].mean())




