import pandas as pd
import scipy.optimize as opt
import warnings
warnings.filterwarnings('ignore')
from scipy.optimize import minimize


df = pd.read_csv('https://raw.githubusercontent.com/LeeMorinUCF/QMB6358F21/main/final_exam/applications.csv')



#### will finsih later and then optimize this using scipy.optimize or summin
#### this is too fun to me and everything else feels like a waste of time
# I want money. Like, everything else right now is time consuming nonsense


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


