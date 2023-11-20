import pandas as pd
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('https://raw.githubusercontent.com/nickkatsy/python_ml_ect_/master/credit_risk.csv')

print(df.info())
print(df.isna().sum())
print(df.nunique())
print(df['Default'].value_counts())


df['Default'] = pd.get_dummies(df['Default'], prefix='Default').iloc[:,0:1]


import matplotlib.pyplot as plt
import seaborn as sns


sns.heatmap(df.corr(),annot=True)
plt.show()


def descriptive_subplots(df):
    plt_, axs = plt.subplots(3,3,figsize=(10,6))
    sns.countplot(df,x='Home',ax=axs[0,0])
    sns.countplot(df,x='Status',ax=axs[0,1])
    sns.scatterplot(df,x='Age',y='Income',ax=axs[0,2])
    sns.violinplot(df[['Status']],ax=axs[1,0])
    sns.countplot(df,x='Default',ax=axs[1,1])
    sns.kdeplot(df,x='Age',ax=axs[1,2])
    sns.countplot(df,x='Intent',ax=axs[2,0])
    sns.scatterplot(df[['Income']],ax=axs[2,1])
    sns.countplot(df,x='Status',ax=axs[2,2])
    plt.show()

descriptive_subplots(df)


df = df.drop('Id',axis=1)


X = df.drop('Default',axis=1)
y = df[['Default']]


from sklearn.model_selection import train_test_split


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)


from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer


ohe = OneHotEncoder()
imp = SimpleImputer()



ct = make_column_transformer(
    (ohe, ['Home','Intent']),
    (imp, ['Emp_length','Rate']),
    remainder='passthrough')


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


from sklearn.pipeline import make_pipeline


clf_pipe = make_pipeline(ct,LogisticRegression()).fit(X_train, y_train)
rfc_pipe = make_pipeline(ct,RandomForestClassifier()).fit(X_train, y_train)
pipe_nb = make_pipeline(ct,GaussianNB()).fit(X_train, y_train)
pipe_gbc = make_pipeline(ct,GradientBoostingClassifier()).fit(X_train, y_train)

# Predictions
clf_pred = clf_pipe.predict(X_test)
clf_pred_prob = clf_pipe.predict_proba(X_test)[::,1]

rfc_pred = rfc_pipe.predict(X_test)
rfc_pred_prob = rfc_pipe.predict_proba(X_test)[::,1]

nb_pred = pipe_nb.predict(X_test)
nb_pred_prob = pipe_nb.predict_proba(X_test)[::,1]

gbc_pred = pipe_gbc.predict(X_test)
gbc_pred_prob = pipe_gbc.predict_proba(X_test)[::,1]

from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score

# results from models
def evaluate_model(model_name,y_true,y_pred,y_pred_prob):
    acc = accuracy_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_pred_prob)
    print(f'{model_name} - Accuracy: {acc * 100:.2f}%, ROC-AUC: {roc * 100:.2f}%')

evaluate_model('Logistic Regression', y_test,clf_pred,clf_pred_prob)
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

roc_curve_plot(y_test,clf_pred_prob,'Logistic Regression')
roc_curve_plot(y_test,rfc_pred_prob,'Random Forest')
roc_curve_plot(y_test,nb_pred_prob,'Naive Bayes')
roc_curve_plot(y_test,gbc_pred_prob,'Gradient Boosting')
plt.legend()
plt.show()
