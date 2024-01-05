import pandas as pd
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('C:/ML/python/data/credit_risk.csv',delimiter=',')

df.info()
df = df.drop('Id',axis=1)

df.describe()
print(df.dtypes)
df.nunique()
df.isna().sum()


df['Default'] = [1 if X == 'Y' else 0 for X in df['Default']]


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df1 = df.copy()


for i in df1:
    df1[i] = le.fit_transform(df1[i])




import matplotlib.pyplot as plt
import seaborn as sns


sns.heatmap(df1.corr(), annot=True)
plt.show()

def sub(df1):
    _,axs = plt.subplots(2,2,figsize=(10,5))
    sns.countplot(x='Default',ax=axs[0,0],data=df1)
    sns.violinplot(x='Status',y='Default',ax=axs[0,1],data=df1)
    sns.boxplot(x='Home',y='Income',ax=axs[1,0],data=df1)
    sns.barplot(x='Default',y='Status',ax=axs[1,1,],data=df1)
    plt.show()


sub(df1)



X = df.drop('Default',axis=1)
y = df['Default']



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)


from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()

from sklearn.impute import SimpleImputer
imp = SimpleImputer()

from sklearn.compose import make_column_transformer

ct = make_column_transformer(
    (ohe, X.select_dtypes(include='object').columns),
    (imp, X.select_dtypes(include=['int64','float64']).columns),
    remainder='passthrough')


ct.fit_transform(X)


from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,BaggingClassifier

rfc = RandomForestClassifier()
gbc = GradientBoostingClassifier()
BC = BaggingClassifier()


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()

from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score,f1_score


def model_metrics(model,X_train,X_test,y_train,y_test):
    pipe = make_pipeline(ct,model).fit(X_train,y_train)
    pred = pipe.predict(X_test)
    pred_prob = pipe.predict_proba(X_test)[:,1]
    acc = accuracy_score(y_test,pred)
    f1 = f1_score(y_test,pred)
    roc = roc_auc_score(y_test,pred_prob)
    print(f'{model.__class__.__name__}, --Accuracy-- {acc*100:.2f}%; --ROC-- {roc*100:.2f}%; --F1-- {f1*100:.2f}%')
    return pred,pred_prob


lr_pred,lr_pred_prob = model_metrics(lr, X_train, X_test, y_train, y_test)
gbc_pred,gbc_pred_prob = model_metrics(gbc, X_train, X_test, y_train, y_test)
rfc_pred,rfc_pred_prob = model_metrics(rfc, X_train, X_test, y_train, y_test)
BC_pred,BC_pred_prob = model_metrics(BC, X_train, X_test, y_train, y_test)
tree_pred,tree_pred_prob = model_metrics(tree, X_train, X_test, y_train, y_test)
nb_pred,nb_pred_prob = model_metrics(nb, X_train, X_test, y_train, y_test)



def ROC(y_test,y_pred_prob,model):
    fpr,tpr, _ = roc_curve(y_test,y_pred_prob)
    plt.plot(fpr,tpr,label=model.__class__.__name__)
    plt.xlabel('False Positive')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')


ROC(y_test,lr_pred_prob,lr)
ROC(y_test,gbc_pred_prob,gbc)
ROC(y_test,rfc_pred_prob,rfc)
ROC(y_test,tree_pred_prob,tree)
ROC(y_test,BC_pred_prob,BC)
ROC(y_test,nb_pred_prob,nb)
plt.legend()
plt.show()
