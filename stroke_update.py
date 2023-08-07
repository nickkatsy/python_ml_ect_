import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

url = 'https://raw.githubusercontent.com/nickkas/python_ml_ect_/master/stroke.csv'

df = pd.read_csv(url)
df.info()

df['gender'].value_counts()
df['gender'] = pd.get_dummies(df.gender,prefix='gender').iloc[:,0:1]

df['id'].value_counts()

df['stroke'].value_counts()

df.isna().sum()
df.nunique


X = df.drop(['id','stroke'],axis=1)
y = df[['stroke']]


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=1)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

clf = LogisticRegression()

rfc = RandomForestClassifier()

knn = KNeighborsClassifier(n_neighbors=3)





from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer

ohe = OneHotEncoder()
imputer = SimpleImputer()

ct = make_column_transformer(
    (ohe,['ever_married','work_type','Residence_type','smoking_status']),
    (imputer,['bmi']),remainder='passthrough')


ct.fit_transform(X)



from sklearn.pipeline import make_pipeline

clf_pipe = make_pipeline(ct,clf).fit(X_train,y_train)
clf_pred = clf_pipe.predict(X_test)
clf_pred_prob = clf_pipe.predict_proba(X_test)[::,1]


rfc_pipe = make_pipeline(ct,rfc).fit(X_train,y_train)
rfc_pred = rfc_pipe.predict(X_test)
rfc_pred_prob = rfc_pipe.predict_proba(X_test)[::,1]


knn_pipe = make_pipeline(ct,knn).fit(X_train,y_train)
knn_pred = knn_pipe.predict(X_test)
knn_pred_prob = knn_pipe.predict_proba(X_test)[::,1]

from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score

#Logistic Regression Results

print('logistic regression accuracy= ',accuracy_score(y_test, clf_pred))
print('Logistic Regression roc= ',roc_auc_score(y_test, clf_pred_prob))

# RandomForrest Classification Results

print('RandomForrest accuracy=',accuracy_score(y_test, rfc_pred))
print('RandomForrest ROC= ',roc_auc_score(y_test, rfc_pred_prob))

# KNN Classification Results

print('KNN accuracy=',accuracy_score(y_test, knn_pred))
print('KNN roc= ',roc_auc_score(y_test, knn_pred_prob))


fpr, tpr, _ = roc_curve(y_test,  clf_pred_prob)
plt.plot(fpr,tpr)
plt.title('ROC Curve For Logistic Regression Model')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
