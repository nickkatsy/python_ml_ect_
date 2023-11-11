import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('https://raw.githubusercontent.com/nickkas/python_ml_ect_/master/cancer.csv')


df.info()

df.isna().sum()

df.nunique()

print(df.dtypes)
df.shape

df = df.drop(['Unnamed: 32','id'],axis=1)

df.corr()



df['diagnosis'] = pd.get_dummies(df.diagnosis,prefix='diagnosis').iloc[:,0:1]

def subplots(df):
    his,ax1 = plt.subplots(4,3,figsize=(10,6))
    sns.histplot(df,x='diagnosis',ax=ax1[0,0])
    sns.histplot(df,x='radius_mean',ax=ax1[0,1])
    sns.histplot(df,x='texture_mean',ax=ax1[0,2])
    sns.histplot(df,x='perimeter_mean',ax=ax1[1,0])
    sns.histplot(df,x='area_mean',ax=ax1[1,1])
    sns.histplot(df,x='smoothness_mean',ax=ax1[1,2])
    sns.histplot(df,x='smoothness_mean',ax=ax1[2,0])
    sns.histplot(df,x='concavity_mean',ax=ax1[2,1])
    sns.histplot(df,x='compactness_mean',ax=ax1[2,2])   
    sns.histplot(df,x='concave points_mean',ax=ax1[3,0])
    sns.histplot(df,x='symmetry_mean',ax=ax1[3,1])
    sns.histplot(df,x='fractal_dimension_mean',ax=ax1[3,2])
    plt.show()



subplots(df)




plt.figure(figsize=(12,5))
sns.displot(df,x='diagnosis')



X = df.drop('diagnosis',axis=1)
y = df[['diagnosis']]

y.value_counts(normalize=True)


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit_transform(X)


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


clf = LogisticRegression().fit(X_train,y_train)
pred_clf = clf.predict(X_test)
pred_prob_clf = clf.predict_proba(X_test)[::,1]


tree = DecisionTreeClassifier().fit(X_train,y_train)
tree_pred = tree.predict(X_test)
tree_pred_prob = tree.predict_proba(X_test)[::,1]

rfc = RandomForestClassifier().fit(X_train,y_train)
pred_rfc = rfc.predict(X_test)
rfc_pred_prob = rfc.predict_proba(X_test)[::,1]



from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score

# Logistic Regression(highest roc out of the three classification methods) Results
Logistic_regression_accuracy = accuracy_score(y_test,pred_clf)
print('accuracy of Logistic Regression model= ',Logistic_regression_accuracy*100)

LogisticRegression_roc = roc_auc_score(y_test, pred_prob_clf)
print('logistic regression roc= ',roc_auc_score(y_test, pred_prob_clf)*100)


# Decision Tree Results

Decisiontree_accuracy = accuracy_score(y_test, tree_pred)
print('Accuracy of Decision Tree= ',Decisiontree_accuracy*100)

Decisiontree_roc = roc_auc_score(y_test, tree_pred_prob)
print('roc for Desision Tree Classification model= ',roc_auc_score(y_test, tree_pred_prob)*100)



# Random Forest Classification Results

#highest ROC score out of the three models

rfc_accuracy = accuracy_score(y_test, pred_rfc)
print('accuarcy of Random forest Classifier= ',roc_auc_score(y_test,pred_rfc)*100)


rfc_roc = roc_auc_score(y_test, rfc_pred_prob)
print('roc for Random Forest Classifier= ', roc_auc_score(y_test, rfc_pred_prob)*100)

def roc_clf(y_true,pred_prob_clf):
    fpr, tpr, _ = roc_curve(y_test, pred_prob_clf)
    plt.plot(fpr,tpr)
    plt.title('ROC Curve For Logistic Regression')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    

roc_clf(y_test,pred_prob_clf)
