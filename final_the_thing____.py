import pandas as pd
import scipy.optimize as opt
import warnings
warnings.filterwarnings('ignore')
from scipy.optimize import minimize







app = pd.read_csv('https://raw.githubusercontent.com/LeeMorinUCF/QMB6358F21/main/final_exam/applications.csv')
credit = pd.read_csv('https://raw.githubusercontent.com/LeeMorinUCF/QMB6358F21/main/final_exam/credit_bureau.csv')


#### will finsih later and then optimize this using scipy.optimize or summin
#### this is too fun to me and everything else feels like a waste of time
# I want money. Like, everything else right now is time consuming nonsense


df = app.copy()

df.info()

df.isna().sum()
df.describe()


df.corr()
df.nunique()

print(df.shape)
print(df.dtypes)
df['homeownership'].value_counts()


# making home ownership a dummy

df['homeownership'] = pd.get_dummies(df.homeownership,prefix='homeownership').iloc[:,0:1]





import seaborn as sns
import matplotlib.pyplot as plt




sns.heatmap(df.corr(), annot=True)

plt.title('Applicant Income')
plt.scatter(df['app_id'],df['income'])
plt.xlabel('app_income')
plt.ylabel('income')





plt.title('Credit Limit Expenditures')
sns.scatterplot(df,x='purchases',y='credit_limit')


X = df.drop(['zip_code','ssn','app_id','homeownership'],axis=1)
y = df[['homeownership']]


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)


from sklearn.linear_model import LogisticRegression

clf = LogisticRegression().fit(X_train,y_train)

clf_pred = clf.predict(X_test)


clf_pred_prob = clf.predict_proba(X_test)[::,1]


from sklearn.metrics import roc_auc_score,accuracy_score

acc_log_reg = accuracy_score(y_test, clf_pred)
print('accuracy for linear regression model',acc_log_reg*100)
roc_log = roc_auc_score(y_test,clf_pred_prob)
print('roc for linear regression',roc_log*100)





## Credit

credit = credit = pd.read_csv('https://raw.githubusercontent.com/LeeMorinUCF/QMB6358F21/main/final_exam/credit_bureau.csv')


credit.info()
credit.isna().sum()
credit.nunique()

credit.isnull()



print(credit.shape)
print(credit.dtypes)


sns.heatmap(credit.corr(), annot=True)


credit.describe()
credit.nunique()

pl_,axs2 = plt.subplots(2,2,figsize=(10,6))
pl__ = sns.histplot(credit[['num_late']],ax=axs2[0,0])
pl__2 = sns.histplot(credit[['past_def']],ax=axs2[0,1])
pl___3 = sns.histplot(credit[['fico']],ax=axs2[1,0])
plt___23 = sns.histplot(credit['ssn'],ax=axs2[1,1])





    
X = credit.drop('past_def',axis=1)
y = credit[['past_def']]



from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression



rfc_credit = RandomForestClassifier().fit(X,y)
knn = KNeighborsClassifier(n_neighbors=3).fit(X,y)
clf_credit = LogisticRegression().fit(X,y)




## credit for logistic regression

credit_pred_clf = clf_credit.predict(X)
credit_pred_prob = clf_credit.predict_proba(X)[::,1]




## KNN

knn_pred_cred = knn.predict(X)
knn_pred_cred_prob = knn.predict_proba(X)[::,1]



credit_rfc_pred = rfc_credit.predict(X)
credit_rfc_pred_prob = rfc_credit.predict_proba(X)[::,1]




### Results for credit

# No random forest, or roc curve right now

from sklearn.metrics import roc_auc_score, accuracy_score

acc_credit_clf = accuracy_score(y, credit_pred_clf)
print('Credit Data accuracy using logistic regression',acc_credit_clf*100)


acc_cred_knn = accuracy_score(y, credit_rfc_pred)

print('Credit Data using Knn',acc_cred_knn*100)




#### Demographics


dem = pd.read_csv('https://raw.githubusercontent.com/LeeMorinUCF/QMB6358F21/main/final_exam/demographic.csv')

 
dem.info()

dem.isna().sum()

plt.title('Average Income')
dennn_ = sns.kdeplot(dem[['avg_income']])

plt.title('Zip Code')
dennn = sns.kdeplot(dem[['zip_code']])


#

plt.title('density')
dennn = sns.kdeplot(dem[['density']])



credit.corr()
dem.corr()
app.corr()




#### Minimization


import scipy.optimize as optimize
import numpy as np


from scipy.optimize import fsolve



