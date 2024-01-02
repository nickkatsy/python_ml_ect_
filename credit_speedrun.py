import pandas as pd
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('https://raw.githubusercontent.com/LeeMorinUCF/QMB6358S22/main/demo_24_Classification/credit.csv')


df.info()
df.isna().sum()
print(df.dtypes)

df.nunique()


df['Default'] = [1 if X == 1 else 0 for X in df['Default']]

import seaborn as sns
import matplotlib.pyplot as plt


df1 = df.copy()

plt.figure(figsize=(12,6))
sns.heatmap(df1.corr(), annot=True)



def sub(df1):
    plt_,axs = plt.subplots(2,2,figsize=(10,6))
    sns.kdeplot(x='duration',y='age',ax=axs[0,0],data=df1,hue='Default')
    sns.countplot(x='Default',ax=axs[0,1],data=df1,hue='job')
    sns.barplot(x='Default',y='age',ax=axs[1,0],hue='status',data=df1)
    sns.barplot(x='job',y='Default',ax=axs[1,1],data=df1)
    plt.show()
    


sub(df1)



X = df.drop('Default',axis=1)
y = df['Default']



from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=0)



from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


from sklearn.compose import make_column_transformer

ct = make_column_transformer(
    (ohe,X.select_dtypes(include='object').columns),(sc,X.select_dtypes(include=['int64']).columns),remainder='passthrough')


X_train_transformed = ct.fit_transform(X_train)
X_test_transformed = ct.transform(X_test)



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

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=8)

from sklearn.pipeline import make_pipeline





from sklearn.metrics import roc_auc_score,roc_curve,f1_score,confusion_matrix,accuracy_score


def evaluate_model(model,X_train_transformed,X_test_transformed,y_train,y_test):
    pipe = make_pipeline(ct,model).fit(X_train,y_train)
    pred = pipe.predict(X_test)
    pred_prob = pipe.predict_proba(X_test)[::,1]
    acc = accuracy_score(y_test, pred)
    roc = roc_auc_score(y_test, pred_prob)
    f1 = f1_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    print('Confusion Matrix',cm)
    print(f'{model.__class__.__name__} --Accuracy-- {acc*100:.2f}%; --ROC-- {roc*100:.2f}%; --F1-- {f1*100:.2f}%')
    return pred,pred_prob

lr_pred,lr_pred_prob = evaluate_model(lr, X_train_transformed, X_test_transformed, y_train, y_test)
rfc_pred,rfc_pred_prob = evaluate_model(rfc,X_test_transformed,X_test_transformed,y_train,y_test)
gbc_pred,gbc_pred_prob =  evaluate_model(gbc, X_train_transformed, X_test_transformed, y_train, y_test)
nb_pred,nb_pred_prob = evaluate_model(nb, X_train_transformed, X_test_transformed, y_train, y_test)
knn_pred,knn_pred_prob = evaluate_model(knn, X_train_transformed, X_test_transformed, y_train, y_test)
tree_pred,tree_pred_prob = evaluate_model(tree, X_train_transformed, X_test_transformed, y_train, y_test)
BC_pred,BC_pred_prob = evaluate_model(BC, X_train_transformed, X_test_transformed, y_train, y_test)


def ROC_Curve(y_test,y_pred_prob,model):
    fpr,tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.plot(fpr,tpr,label=model.__class__.__name__)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')



ROC_Curve(y_test,lr_pred_prob,lr)
ROC_Curve(y_test,rfc_pred_prob,rfc)
ROC_Curve(y_test,gbc_pred_prob,gbc)
ROC_Curve(y_test,tree_pred_prob,tree)
ROC_Curve(y_test,nb_pred_prob,nb)
ROC_Curve(y_test, knn_pred_prob,knn)
ROC_Curve(y_test,BC_pred_prob,BC)
plt.legend()
plt.show()


# Cross-Validation

from sklearn.model_selection import cross_val_score


def cross_validation(model,X_train_transformed,y_train):
    model = model.fit(X_train_transformed,y_train)
    cv_score = cross_val_score(model, X_train_transformed,y_train,cv=5,scoring='roc_auc').mean()
    print(f'{model.__class__.__name__}, --5-fold Cross-Validation Score-- {cv_score*100:.2f}%')
    return cv_score
    

lr_score = cross_validation(lr, X_train_transformed, y_train)
gbc_score = cross_validation(gbc, X_train_transformed, y_train)
rfc_score = cross_validation(rfc, X_train_transformed, y_train)
tree_score = cross_validation(tree, X_train_transformed, y_train)
nb_score = cross_validation(nb, X_train_transformed, y_train)
knn_score = cross_validation(knn, X_train_transformed, y_train)
BC_score = cross_validation(BC, X_train_transformed, y_train)
