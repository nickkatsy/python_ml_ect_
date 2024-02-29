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


df['Sex'] = df['Sex'].map({'female':0,'male':1})


copy = df.copy()



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
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


lr = LogisticRegression()

rfc = RandomForestClassifier()

knn = KNeighborsClassifier(n_neighbors=13)

tree_clf = DecisionTreeClassifier()

gb = GradientBoostingClassifier()

nb = GaussianNB()
BC = BaggingClassifier()


from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer

ohe = OneHotEncoder()

imp = SimpleImputer(strategy='mean')


ct = make_column_transformer(
    (ohe,X.select_dtypes(include='object').columns),
    (imp,X.select_dtypes(include=['int64','float64']).columns),remainder='passthrough')



ct.fit_transform(X)


from sklearn.pipeline import make_pipeline


from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve

def evaluate_model(X_train,X_test,y_train,y_test,model):
    pipe = make_pipeline(ct,model).fit(X_train,y_train)
    pred = pipe.predict(X_test)
    pred_prob = pipe.predict_proba(X_test)[:,1]
    acc = accuracy_score(y_test,pred)
    roc = roc_auc_score(y_test, pred_prob)
    print(f'{model} - Accuracy: {acc * 100:.2f}%, ROC-AUC: {roc * 100:.2f}%')
    return pred,pred_prob




lr_pred,lr_pred_prob = evaluate_model(X_train, X_test, y_train, y_test, lr)
rfc_pred,rfc_pred_prob = evaluate_model(X_train, X_test, y_train, y_test, rfc)
knn_pred,knn_pred_prob = evaluate_model(X_train, X_test, y_train, y_test, knn)
gb_pred,gb_pred_prob = evaluate_model(X_train, X_test, y_train, y_test, gb)
nb_pred,nb_pred_prob = evaluate_model(X_train, X_test, y_train, y_test, nb)
BC_pred,BC_pred_prob = evaluate_model(X_train, X_test, y_train, y_test, BC)




def roc_curve_plot(y_true, y_pred_prob, model):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    plt.plot(fpr,tpr,label=model.__class__.__name__)
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

roc_curve_plot(y_test,lr_pred_prob,lr)
roc_curve_plot(y_test,rfc_pred_prob,rfc)
roc_curve_plot(y_test,knn_pred_prob,knn)
roc_curve_plot(y_test,nb_pred_prob,nb)
roc_curve_plot(y_test,gb_pred_prob,gb)
plt.legend()
plt.show()


