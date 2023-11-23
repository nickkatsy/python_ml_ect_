import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('https://raw.githubusercontent.com/LeeMorinUCF/QMB6358S22/main/demo_24_Classification/credit.csv')


df.info()

df.describe()
df.isna().sum()

df.nunique()

import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(df.corr(), annot=True)


def count(df):
    plt__,axs2 = plt.subplots(2,2,figsize=(10,6))
    sns.boxplot(df[['amount']],ax=axs2[0,0])
    sns.boxplot(df[['installment']],ax=axs2[0,1])
    sns.violinplot(df[['Default']],ax=axs2[1,0])
    sns.boxplot(df[['cards']],ax=axs2[1,1])
    
    plt.show()


count(df)



X = df.drop('Default',axis=1)
y = df[['Default']]

from sklearn.model_selection import train_test_split


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)


from sklearn.linear_model import LogisticRegression



clf = LogisticRegression()


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()


from sklearn.tree import DecisionTreeClassifier

trees = DecisionTreeClassifier()

from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier()

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer


ohe = OneHotEncoder()


ct = make_column_transformer(
    (ohe,['employ','checkingstatus1','history','purpose','savings','status','others','property','otherplans','housing','job','tele','foreign']),remainder='passthrough')



ct.fit_transform(X)


from sklearn.pipeline import make_pipeline

# pipeline for logistic regression
clf_pipe = make_pipeline(ct,clf).fit(X_train,y_train)

pipe_pred = clf_pipe.predict(X_test)

pipe_pred_prob = clf_pipe.predict_proba(X_test)[::,1]


# Pipeline for Random Forest

rfc_pipe = make_pipeline(ct,rfc).fit(X_train,y_train)
rfc_pred = rfc_pipe.predict(X_test)
rfc_pred_prob = rfc_pipe.predict_proba(X_test)[::,1]


#pipeline for Decision tree classifier

tree_pipe = make_pipeline(ct,trees).fit(X_train,y_train)
tree_pred = tree_pipe.predict(X_test)
tree_pred_prob = tree_pipe.predict_proba(X_test)[::,1]


# pipeline for Gradient Boost Classifier

gbc_pipe = make_pipeline(ct,gb).fit(X_train,y_train)
gbc_pred = gbc_pipe.predict(X_test)
gbc_pred_prob = gbc_pipe.predict_proba(X_test)[::,1]


# Naive Bayes Pipeline

nb_pipe = make_pipeline(ct,nb).fit(X_train,y_train)
nb_pred = nb_pipe.predict(X_test)
nb_pred_prob = nb_pipe.predict_proba(X_test)[::,1]



from sklearn.metrics import accuracy_score,f1_score,roc_curve,roc_auc_score


# Results from using logistic regression classification
F1_score_clf = f1_score(y_test, pipe_pred)
print(F1_score_clf)

acc_clf = accuracy_score(y_test, pipe_pred)
print(f'Logistic Regression Score: {acc_clf}')

roc_clf = roc_auc_score(y_test, pipe_pred_prob)
print('ROC score for',roc_clf*100)


# Results from Random Forest Classifier

acc_rfc = accuracy_score(y_test, rfc_pred)
print(f'accuracy using Random Forest: {acc_rfc}')
roc_rfc = roc_auc_score(y_test, rfc_pred_prob)
print(f'roc_auc score using Random Forest Classification: {roc_rfc}')


# Results from using Decision Tree Classifier

acc_trees = accuracy_score(y_test, tree_pred)
print(f'Accuracy of Decision Tree Classifier: {acc_trees}')
roc_trees = roc_auc_score(y_test, tree_pred_prob)
print(f'ROC_auc score using Decsion Trees: {roc_trees}')


# Results using Naive Bayes

acc_nb = accuracy_score(y_test, nb_pred)
print(f'Accuracy of Naive Bayes Classifier: {acc_nb}')

roc_nb = roc_auc_score(y_test, nb_pred_prob)
print(f'roc_auc using Naive Bayes Classifier: {roc_nb}')


# Results from Gradient Boost
acc_gb = accuracy_score(y_test, gbc_pred)
print(f'Accuracy using Gradient Boost: {acc_nb}')

roc_gb = roc_auc_score(y_test, gbc_pred_prob)
print(f'roc_auc using Gradient Boosting Classifier: {roc_gb}')

def evaluate_model(model_name,y_true,y_pred,y_pred_prob):
    acc = accuracy_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_pred_prob)
    print(f'{model_name} - Accuracy: {acc * 100:.2f}%, ROC-AUC: {roc * 100:.2f}%')

evaluate_model('Logistic Regression', y_test,pipe_pred,pipe_pred_prob)
evaluate_model('Random Forest', y_test,rfc_pred,rfc_pred_prob)
evaluate_model('Naive Bayes', y_test,nb_pred,nb_pred_prob)
evaluate_model('Gradient Boosting',y_test,gbc_pred,gbc_pred_prob)

# ROC Curves
def roc_curve_plot(y_true, y_pred_prob, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    plt.plot(fpr,tpr,label=model_name)
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

roc_curve_plot(y_test,pipe_pred_prob,'Logistic Regression')
roc_curve_plot(y_test,rfc_pred_prob,'Random Forest')
roc_curve_plot(y_test,tree_pred_prob,'Decision Trees')
roc_curve_plot(y_test,nb_pred_prob,'Naive Bayes')
roc_curve_plot(y_test,gbc_pred_prob,'Gradient Boosting')
plt.legend()
plt.show()
