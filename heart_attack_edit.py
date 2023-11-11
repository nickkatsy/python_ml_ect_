import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('https://raw.githubusercontent.com/nickkas/python_ml_ect_/master/heart.csv')

df.info()

df.isna().sum()

print(df.shape)

df.dtypes

df['heart_attack'] = df['target']

df = df.drop('target',axis=1)

df.corr()

import seaborn as sns
import matplotlib.pyplot as plt

def subplots(df):
    plt_,ax1 = plt.subplots(3,3,figsize=(12,5))
    sns.histplot(df[['age']],ax=ax1[0,0])
    sns.violinplot(df[['cp']],ax=ax1[0,1])  
    sns.histplot(df[['trestbps']],ax=ax1[0,2])
    sns.histplot(df[['fbs']],ax=ax1[1,0])
    sns.histplot(df[['sex']],ax=ax1[1,1])
    sns.histplot(df['exang'],ax=ax1[1,2])
    sns.histplot(df[['heart_attack']],ax=ax1[2,0])
    sns.histplot(df[['chol']],ax=ax1[2,1])
    sns.kdeplot(df[['oldpeak']],ax=ax1[2,2])
    plt.show()



subplots(df)


import statsmodels.api as sm

model_sex = sm.GLM(exog=sm.add_constant(df[['sex']]),endog=df[['heart_attack']]).fit()
print(model_sex.summary())


model_age = sm.OLS(exog=sm.add_constant(df['age']),endog=df[['heart_attack']]).fit()
print(model_age.summary())

model_chol = sm.OLS(exog=sm.add_constant(df[['chol']]),endog=df[['heart_attack']]).fit()
print(model_chol.summary())



X = df.drop('heart_attack',axis=1)
y = df[['heart_attack']]

y.value_counts(normalize=True)


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit_transform(X)




from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)



from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


clf = LogisticRegression().fit(X_train,y_train)
clf_pred = clf.predict(X_test)
clf_pred_prob = clf.predict_proba(X_test)[::,1]

rf = RandomForestClassifier().fit(X_train,y_train)
rf_pred = rf.predict(X_test)
rf_pred_prob = rf.predict_proba(X_test)[::,1]



from sklearn.metrics import roc_auc_score,accuracy_score,roc_curve

#Logistic Regression

clf_acc = accuracy_score(y_test, clf_pred)
print('logistic regression accuracy=', clf_acc*100)

clf_roc = roc_auc_score(y_test, clf_pred_prob)
print('roc for logistic regression model= ',clf_roc*100)


# Random Forest Classification Results

rff_acc = accuracy_score(y_test, rf_pred)
print('Random Forest accuracy= ',rff_acc*100)


rff_roc = roc_auc_score(y_test,rf_pred_prob)
print('Random Forest roc= ',rff_roc*100)


def roc_rfc(y_true,rf_pred_prob):
    fpr,tpr, _ = roc_curve(y_test,rf_pred_prob)
    plt.plot(fpr,tpr)
    plt.title('ROC curve for random forest')
    plt.ylabel('True positive rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    

roc_rfc(y_test, rf_pred_prob)
