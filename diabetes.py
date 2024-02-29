import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('https://raw.githubusercontent.com/nickkas/python_ml_ect_/master/diabetes.csv')


df.info()
df.isna().sum()
df.nunique()
df.dtypes





def desc_plots(df):
    plt_,axs = plt.subplots(2,4,figsize=(10,6))
    sns.boxplot(df[['Pregnancies']],ax=axs[0,0])
    sns.boxplot(df[['Glucose']],ax=axs[0,1])
    sns.boxplot(df[['BloodPressure']],ax=axs[0,2])
    sns.boxplot(df[['SkinThickness']],ax=axs[0,3])
    sns.boxplot(df[['Insulin']],ax=axs[1,0])
    sns.boxplot(df[['BMI']],ax=axs[1,1])
    sns.boxplot(df[['DiabetesPedigreeFunction']],ax=axs[1,2])
    sns.boxplot(df[['Age']],ax=axs[1,3])
    plt.show()



desc_plots(df)


plt.figure(figsize=(12,4))
sns.heatmap(df.corr(),annot=True)







X = df.drop('Outcome',axis=1)
y = df[['Outcome']]

y.value_counts(normalize=True)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=1)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
svc = SVC(probability=True)



lr = LogisticRegression()

rfc = RandomForestClassifier()
Knn = KNeighborsClassifier(n_neighbors=7)

nb = GaussianNB()
BC = BaggingClassifier()
gbc = GradientBoostingClassifier()


from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score,f1_score,confusion_matrix
from sklearn.model_selection import cross_val_score


def evaluate_model(X_train_scaled,X_test_scaled,y_train,y_test,model):
    model = model.fit(X_train_scaled,y_train)
    pred = model.predict(X_test_scaled)
    pred_prob = model.predict_proba(X_test_scaled)[:,1]
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    con = confusion_matrix(y_test, pred)
    roc = roc_auc_score(y_test, pred_prob)
    print(f'{model.__class__.__name__}, --ACC-- {acc*100:.2f}%; --ROC-- {roc*100:.2f}%; --f1-- {f1*100:.2f}%')
    print('confusion matrix',con)
    return pred,pred_prob


lr_pred,lr_pred_prob = evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test, lr)
rfc_pred,rfc_pred_prob = evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test, rfc)
nb_pred,nb_pred_prob = evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test, nb)
BC_pred,BC_pred_prob = evaluate_model(X_train, X_test, y_train, y_test,BC)
gbc_pred,gbc_pred_prob = evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test,gbc)
Knn_pred,Knn_pred_prob = evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test,Knn)
svc_pred,svc_pred_prob = evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test, svc)


def ROC_Curve(y_test,y_pred_prob,model):
    fpr,tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.plot(fpr,tpr,label=model.__class__.__name__)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')



ROC_Curve(y_test,lr_pred_prob,lr)
ROC_Curve(y_test,rfc_pred_prob,rfc)
ROC_Curve(y_test,gbc_pred_prob,gbc)
ROC_Curve(y_test,nb_pred_prob,nb)
ROC_Curve(y_test, Knn_pred_prob,Knn)
ROC_Curve(y_test,BC_pred_prob,BC)
ROC_Curve(y_test, svc_pred_prob, svc)
plt.legend()
plt.show()



#Cross-validation

def cross_validation(model,X,y):
    model = model.fit(X_train_scaled,y_train)
    cv_score = cross_val_score(model, X,y,cv=10,scoring='roc_auc').max()
    print(f'{model.__class__.__name__}, --10-fold Cross-Validation Score-- {cv_score*100:.2f}%')
    return cv_score
    


lr_score = cross_validation(lr, X, y)
gbc_score = cross_validation(gbc,X,y)
rfc_score = cross_validation(rfc, X, y)
nb_score = cross_validation(nb, X, y)
knn_score = cross_validation(Knn, X, y)
BC_score = cross_validation(BC,X,y)
svc_score = cross_validation(svc, X, y)
