import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/nickkas/python_ml_ect_/master/hmda.txt'

df = pd.read_csv(url,delimiter='\t')

df.info()

df.isna().sum()
df.isnull().sum()

df.rename(columns={'s5':'occupancy','s7':'approve','s11':'county','s13':'race',
                   's15':'sex','s17':'income','s23a':'married','s27a':'self_employed',
                   's33':'purchase_price','s34':'other_financing','s35':'liquid_assets',
                   's40':'credit_history','s42':'chmp','s43':'chcp','s44':'chpr',
                   's45':'debt_to_expense','s46':'di_ratio','s50':'appraisal',
                   's53':'pmi_denied','netw':'net_worth','uria':'unemployment',
                   'school':'education','s56':'unverifiable',
                   's52':'pmi_sought'},inplace=True)





df['approve'] = [1 if X == 3 else 0 for X in df['approve']]
df['race'] = [0 if X == 3 else 1 for X in df['race']]
df['married'] = [1 if X == 'M' else 0 for X in df['married']]
df['sex'] = [1 if X == 1 else 0 for X in df['sex']]
df['credit_history'] = [1 if X == 1 else 0 for X in df['credit_history']]

features = ['occupancy','race','sex','income','married','credit_history','di_ratio',
            'pmi_denied','unverifiable','pmi_sought']

X = df[features]
y = df.approve

y.value_counts(normalize=True)


import statsmodels.api as sm

model = sm.OLS(y,X).fit()
print(model.summary())

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(1)
poly_X = poly.fit_transform(X)


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=1)

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X_train,y_train)

pred = clf.predict(X_test)



from sklearn.metrics import roc_auc_score,accuracy_score,roc_curve

print('accuracy=',accuracy_score(y_test, pred))


y_pred = clf.predict_proba(X_test)[::,1]


print('roc=',roc_auc_score(y_test, y_pred))

from sklearn.model_selection import cross_val_score

cv_roc = cross_val_score(clf, poly_X,y,cv=5,scoring='roc_auc').max()
print(cv_roc*100)

fpr, tpr, _ = roc_curve(y_test, y_pred)



plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
