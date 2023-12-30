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
    (ohe, X.select_dtypes(include='object').columns),
    (imp, X.select_dtypes(include=['int64','float64']).columns),
    remainder='passthrough')


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


lr = LogisticRegression()
nb = GaussianNB()
rfc = RandomForestClassifier()
gbc = GradientBoostingClassifier()


from sklearn.pipeline import make_pipeline



# Predictions


from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score,f1_score

# results from models
def evaluate_model(model,X_train,X_test,y_train,y_test):
    pipe = make_pipeline(ct,model).fit(X_train,y_train)
    pred = pipe.predict(X_test)
    pred_prob = pipe.predict_proba(X_test)[::,1]
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc = roc_auc_score(y_test,pred_prob)
    print(f'{model.__class__.__name__} - Accuracy: {acc * 100:.2f}%, ROC-AUC: {roc * 100:.2f}%; --F1--{f1*100:.2f}%')
    return pred,pred_prob

lr_pred,lr_pred_prob = evaluate_model(lr, X_train, X_test, y_train, y_test)
rfc_pred,rfc_pred_prob = evaluate_model(rfc, X_train, X_test, y_train, y_test)
nb_pred,nb_pred_prob = evaluate_model(nb, X_train, X_test, y_train, y_test)
gbc_pred,gbc_pred_prob = evaluate_model(gbc, X_train, X_test, y_train, y_test)

# ROC Curves
def roc_curve_plot(y_test,y_pred_prob, model):
    fpr, tpr, _ = roc_curve(y_test,y_pred_prob)
    plt.plot(fpr,tpr,label=model.__class__.__name__)
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

roc_curve_plot(y_test,lr_pred_prob,lr)
roc_curve_plot(y_test,rfc_pred_prob,rfc)
roc_curve_plot(y_test,nb_pred_prob,nb)
roc_curve_plot(y_test,gbc_pred_prob,gbc)
plt.legend()
plt.show()
