import pandas as pd
import warnings
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('https://raw.githubusercontent.com/LeeMorinUCF/QMB6358F23/main/demo_14_linear_models_in_python/credit_data.csv')

df.info()



df.isna().sum()

df.isnull().sum()


import seaborn as sns
import matplotlib.pyplot as plt



sns.heatmap(df.corr(),annot=True)



plt__,axs = plt.subplots(2,2,figsize=(10,6))
plt___ = sns.scatterplot(df[['amount']],ax=axs[0,0])
plt___ = sns.violinplot(df[['AA']],ax=axs[0,1])
plt___ = sns.histplot(df[['D']],ax=axs[1,0])
plt_____ = sns.violinplot(df[['B']],ax=axs[1,1])




#### risk of default

X = df[['bmaxrate','amount','close']]
y = df[['default']]



from sklearn.model_selection import train_test_split


X_train_,X_test_,y_train_,y_test_ = train_test_split(X,y,test_size=.20,random_state=42)



from sklearn.linear_model import LogisticRegression

clf = LogisticRegression().fit(X_train_,y_train_)

clf_pred = clf.predict(X_test_)

clf_pred_prob = clf.predict_proba(X_test_)[::,1]



from sklearn.ensemble import RandomForestClassifier


rfc = RandomForestClassifier().fit(X_train_,y_train_)

rfc_pred = rfc.predict(X_test_)

rfc_pred_prob = rfc.predict_proba(X_test_)[::,1]


from sklearn.metrics import roc_auc_score,accuracy_score



acc_rfc = accuracy_score(y_test_, rfc_pred)
print('Accuracy using random forrest',acc_rfc*100)




rfc_roc = roc_auc_score(y_test_, rfc_pred_prob)
print('roc for random forrest',roc_auc_score(y_test_,rfc_pred_prob))


acc_clf = accuracy_score(y_test_,clf_pred)
print(' Logistic Regression',accuracy_score(y_test_, clf_pred))

roc_clf = roc_auc_score(y_test_, clf_pred_prob)
print('Rpc for Logisitc regression',roc_auc_score(y_test_, clf_pred_prob))



#### So bad 

X_amount = df.drop('amount',axis=1)
y_amount = df[['amount']]


from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures()

poly_amount = poly.fit_transform(X_amount)

from sklearn.linear_model import LinearRegression

lr = LinearRegression().fit(X_amount,y_amount)

lr_score = lr.predict(X_amount)


from sklearn.model_selection import cross_val_score

cv = cross_val_score(lr, poly_amount,y_amount,cv=10,scoring='r2').mean()
print('10-fold_poly',cv*100)















