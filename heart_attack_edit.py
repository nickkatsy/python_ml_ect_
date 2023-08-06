import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('C:/ml/python/projects/heart.csv',delimiter=',')

df.info()

df.isna().sum()

print(df.shape)

df.dtypes

df['heart_attack'] = df['target']

df = df.drop('target',axis=1)

df.corr()

import seaborn as sns
import matplotlib.pyplot as plt

plt_,ax1 = plt.subplots(3,3,figsize=(12,5))
plt_1 = sns.histplot(df[['age']],ax=ax1[0,0])
plt_2 = sns.violinplot(df[['cp']],ax=ax1[0,1])
plt_3 = sns.histplot(df[['trestbps']],ax=ax1[0,2])
plt_4 = sns.histplot(df[['fbs']],ax=ax1[1,0])
plt_5 = sns.histplot(df[['sex']],ax=ax1[1,1])
plt_6 = sns.histplot(df['exang'],ax=ax1[1,2])
plt_7 = sns.histplot(df[['heart_attack']],ax=ax1[2,0])
plt_8 = sns.histplot(df[['chol']],ax=ax1[2,1])
plt_9 = sns.kdeplot(df[['oldpeak']],ax=ax1[2,2])



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



from sklearn.metrics import roc_auc_score,accuracy_score

#Logistic Regression

clf_acc = accuracy_score(y_test, clf_pred)
print('logistic regression accuracy=', clf_acc*100)

clf_roc = roc_auc_score(y_test, clf_pred_prob)
print('roc for logistic regression model= ',clf_roc*100)


# RandomForrest Classification Results

rff_acc = accuracy_score(y_test, rf_pred)
print('Random Forrest accuracy= ',rff_acc*100)


rff_roc = roc_auc_score(y_test,rf_pred_prob)
print('RandomForrest roc= ',rff_roc*100)
