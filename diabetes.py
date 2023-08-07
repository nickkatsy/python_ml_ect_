import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('C:/ml/python/data/diabetes.csv',delimiter=',')


df.info()
df.isna().sum()
df.nunique()
df.dtypes


df.corr()

plt_,axs = plt.subplots(2,4,figsize=(10,6))
plt_ = sns.boxplot(df[['Pregnancies']],ax=axs[0,0])
plt_2 = sns.boxplot(df[['Glucose']],ax=axs[0,1])
plt_3 = sns.boxplot(df[['BloodPressure']],ax=axs[0,2])
plt_4 = sns.boxplot(df[['SkinThickness']],ax=axs[0,3])
plt_5 = sns.boxplot(df[['Insulin']],ax=axs[1,0])
plt_6 = sns.boxplot(df[['BMI']],ax=axs[1,1])
plt_7 = sns.boxplot(df[['DiabetesPedigreeFunction']],ax=axs[1,2])
plt_8 = sns.boxplot(df[['Age']],ax=axs[1,3])



plt.figure(figsize=(12,4))
sns.heatmap(df.corr(),annot=True)


X = df.drop('Outcome',axis=1)
y = df[['Outcome']]

y.value_counts(normalize=True)



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit_transform(X)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


clf = LogisticRegression().fit(X,y)
clf_pred = clf.predict(X)
clf_pred_prob = clf.predict_proba(X)[::,1]

rfc = RandomForestClassifier().fit(X,y)
rfc_pred = rfc.predict(X)
rfc_pred_prob = rfc.predict_proba(X)[::,1]

Knn = KNeighborsClassifier(n_neighbors=6).fit(X,y)
Knn_pred = Knn.predict(X)
Knn_pred_prob = Knn.predict_proba(X)[::,1]


from sklearn.metrics import roc_auc_score,accuracy_score, roc_curve
from sklearn.model_selection import cross_val_score

# Logistic Regression Results
clf_acc = accuracy_score(y, clf_pred)
print('Accuracy of Logistic Regression Model',clf_acc*100)

clf_roc = roc_auc_score(y, clf_pred_prob)
print('logistic Regression ROC: ',clf_roc*100)

#Logistic Regression Model produces a higher roc with 10 fold cross-validation
cv_clf = cross_val_score(clf, X,y,cv=10,scoring='roc_auc').max()
print('Logistic Regression ROC with Cross-Validation',cv_clf*100)


#RandomForrest Classification Results

acc_rfc = accuracy_score(y,rfc_pred)
print('accuracy score RandomForrest Classification',acc_rfc*100)

roc_rfc = roc_auc_score(y,rfc_pred_prob)
print('roc using RandomForrest Classification= ',roc_rfc)

# KNN Classification Results

acc_knn = accuracy_score(y, Knn_pred)
print('accuarcy score using knn= ',acc_knn*100)

roc_knn = roc_auc_score(y, Knn_pred_prob)
print('roc using knn',roc_auc_score(y, Knn_pred_prob)*100)



fpr, tpr, _ = roc_curve(y,rfc_pred_prob)
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()




