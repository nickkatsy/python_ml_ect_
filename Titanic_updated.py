import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('http://bit.ly/kaggletrain')



df.info()

df.isna().sum()

df.nunique()

print(df.dtypes)

df = df.drop((['Name','PassengerId','Cabin','Ticket']),axis=1)

copy = df.copy()

copy.dropna()


copy['Sex'] = copy['Sex'].map({'female':0,'male':1})


def misc(copy):
    plt_,ax1 = plt.subplots(2,3,figsize=(10,6))
    sns.kdeplot(copy[['Fare']],ax=ax1[0,0])
    sns.histplot(copy[['Survived']],ax=ax1[0,1])
    sns.violinplot(copy[['Pclass']],ax=ax1[0,2])
    sns.boxplot(copy[['SibSp']],ax=ax1[1,0])
    sns.histplot(copy[['Age']],ax=ax1[1,1])
    sns.histplot(copy[['Sex']],ax=ax1[1,2])
    plt.show()


misc(copy)



def more_subs(copy):
    plt_2,ax2 = plt.subplots(2,2,figsize=(10,6))
    sns.barplot(copy,x='Sex',y='Survived',ax=ax2[0,0])
    sns.boxplot(copy,x='Survived',y='Age',ax=ax2[0,1])
    sns.histplot(copy[['Age']],ax=ax2[1,0])
    sns.barplot(copy,x='Sex',y='Fare',ax=ax2[1,1])
    plt.show()



more_subs(copy)



X = df.drop('Survived',axis=1)
y = df[['Survived']]

y.value_counts(normalize=True)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.30,random_state=42)



from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

clf = LogisticRegression()

rfc = RandomForestClassifier()

knn = KNeighborsClassifier(n_neighbors=13)

tree_clf = DecisionTreeClassifier()


from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer

ohe = OneHotEncoder()

imp = SimpleImputer(strategy='mean')


ct = make_column_transformer((ohe,['Pclass','Embarked','Sex']),(imp,['Age']),remainder='passthrough')



ct.fit_transform(X)


from sklearn.pipeline import make_pipeline

pipe_clf = make_pipeline(ct,clf).fit(X_train,y_train)
pred_clf = pipe_clf.predict(X_test)
pred_prob_clf = pipe_clf.predict_proba(X_test)[::,1]

pipe_rfc = make_pipeline(ct,rfc).fit(X_train,y_train)
pred_rfc = pipe_rfc.predict(X_test)
rfc_pred_prob = pipe_rfc.predict_proba(X_test)[::,1]

pipe_knn = make_pipeline(ct,knn).fit(X_train,y_train)
pred_knn = pipe_knn.predict(X_test)
knn_pred_prob = pipe_knn.predict_proba(X_test)[::,1]


pipe_tree = make_pipeline(ct,tree_clf).fit(X_train,y_train)
pred_tree = pipe_tree.predict(X_test)
pred_prob_tree = pipe_tree.predict_proba(X_test)[::,1]

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

#logistic regression results

acc = accuracy_score(y_test, pred_clf)
print('accuracy score for logistic regression= ',acc*100)
print('ROC for logistic regression model= ',roc_auc_score(y_test, pred_prob_clf))



#Random Forest Classification Results

print('accuracy for Random Forest classifier= ',roc_auc_score(y_test, pred_rfc))
print('roc_auc for Random Forest classifier= ',roc_auc_score(y_test, rfc_pred_prob))

# KNN Classification results

print('accuracy score for KNN classifier',accuracy_score(y_test, pred_knn))
print('roc_auc for knn classification',roc_auc_score(y_test,knn_pred_prob))

# Decision Tree Classfication results
print('Accuracy for decsion tree classifier= ',accuracy_score(y_test,pred_tree))
print('ROC for tree classifier',roc_auc_score(y_test, pred_prob_tree))


def logistic_roc(y_test,pred_prob_clf):
    fpr,tpr, _ = roc_curve(y_test,pred_prob_clf)
    plt.plot(fpr,tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('The ROC curve from the Logistic Regression Model')
    plt.show()
    

logistic_roc(y_test, pred_prob_clf)
