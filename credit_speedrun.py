import pandas as pd
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('https://raw.githubusercontent.com/LeeMorinUCF/QMB6358S22/main/demo_24_Classification/credit.csv')


df.info()
df.isna().sum()
print(df.dtypes)

df.nunique()


import seaborn as sns
import matplotlib.pyplot as plt


df1 = df.copy()

plt.figure(figsize=(12,6))
sns.heatmap(df1.corr(), annot=True)



def sub(df1):
    plt_,axs = plt.subplots(2,2,figsize=(10,6))
    sns.histplot(x='duration',y='age',ax=axs[0,0],data=df1,hue='Default')
    sns.countplot(x='Default',ax=axs[0,1],data=df1,hue='job')
    sns.barplot(x='Default',y='age',ax=axs[1,0],hue='status',data=df1)
    sns.violinplot(x='Default',y='job',ax=axs[1,1],data=df1)
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


ct.fit_transform(X_train)
ct.transform(X_test)



from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

rfc = RandomForestClassifier()
gbc = GradientBoostingClassifier()


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=6)

from sklearn.pipeline import make_pipeline

lr_pipe = make_pipeline(ct,lr).fit(X_train,y_train)
lr_pred = lr_pipe.predict(X_test)
lr_pred_prob = lr_pipe.predict_proba(X_test)[::,1]


rfc_pipe = make_pipeline(ct,rfc).fit(X_train,y_train)
rfc_pred = rfc_pipe.predict(X_test)
rfc_pred_prob = rfc_pipe.predict_proba(X_test)[::,1]


gbc_pipe = make_pipeline(ct,gbc).fit(X_train,y_train)
gbc_pred = gbc_pipe.predict(X_test)
gbc_pred_prob = gbc_pipe.predict_proba(X_test)[::,1]


tree_pipe = make_pipeline(ct,tree).fit(X_train,y_train)
tree_pred = tree_pipe.predict(X_test)
tree_pred_prob = tree_pipe.predict_proba(X_test)[::,1]

nb_pipe = make_pipeline(ct,nb).fit(X_train,y_train)
nb_pred = nb_pipe.predict(X_test)
nb_pred_prob = nb_pipe.predict_proba(X_test)[::,1]


knn_pipe = make_pipeline(ct,knn).fit(X_train,y_train)
knn_pred = knn_pipe.predict(X_test)
knn_pred_prob = knn_pipe.predict_proba(X_test)[::,1]



from sklearn.metrics import roc_auc_score,roc_curve,f1_score,confusion_matrix,accuracy_score


def evaluate_model(y_true,y_pred,y_pred_prob,model_name):
    acc = accuracy_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_pred_prob)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print('Confusion Matrix',cm)
    print(f'{model_name} --Accuracy-- {acc*100:.2f}%; --ROC-- {roc*100:.2f}%; --F1-- {f1*100:.2f}%')
    

evaluate_model(y_test, lr_pred, lr_pred_prob, 'Logistic Regression')
evaluate_model(y_test, rfc_pred, rfc_pred_prob, 'Random Forest')
evaluate_model(y_test, tree_pred, tree_pred_prob, 'Decision Tree Classifier')
evaluate_model(y_test, gbc_pred, gbc_pred_prob, 'Gradient Boost')
evaluate_model(y_test, knn_pred, knn_pred_prob, 'K-Nearest Neighbors')
evaluate_model(y_test, nb_pred, nb_pred_prob, 'Naive Bayes')


def ROC_Curve(y_true,y_pred_prob,model_name):
    fpr,tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.plot(fpr,tpr,label=model_name)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')



ROC_Curve(y_test, lr_pred_prob,'Logistic Regression')
ROC_Curve(y_test, rfc_pred_prob, 'Random Forest')
ROC_Curve(y_test, gbc_pred_prob, 'Gradient Boost Classifier')
ROC_Curve(y_test, tree_pred_prob, 'Trees')
ROC_Curve(y_test, knn_pred_prob, 'K-Nearest Neighbors')
ROC_Curve(y_test, nb_pred_prob, 'Naive Bayes')
plt.legend()
plt.show()
