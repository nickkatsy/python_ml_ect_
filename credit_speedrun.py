import pandas as pd
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('https://raw.githubusercontent.com/LeeMorinUCF/QMB6358S22/main/demo_24_Classification/credit.csv')
df.info()
df.isna().sum()
print(df.dtypes)


df1 = df.copy()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for i in df1:
    df1[i] = le.fit_transform(df1[i])



import seaborn as sns
import matplotlib.pyplot as plt


def sub(df1):
    f,axs = plt.subplots(2,2,figsize=(10,6))
    sns.barplot(x='job',y='Default',ax=axs[0,0],data=df1)
    sns.barplot(x='installment',y='Default',ax=axs[0,1],data=df1)
    sns.scatterplot(x='amount',y='age',ax=axs[1,0],data=df1)
    sns.barplot(x='employ',y='Default',ax=axs[1,1],data=df1)
    plt.show()
    
sub(df1)


X = df.drop('Default',axis=1)
y = df['Default']


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)

from sklearn.preprocessing import OneHotEncoder,StandardScaler
ohe = OneHotEncoder()
sc = StandardScaler()

from sklearn.compose import make_column_transformer

ct = make_column_transformer(
    (ohe,X.select_dtypes(include='object').columns),
    (sc,X.select_dtypes(include=['int64','float64']).columns),remainder='passthrough')



ct.fit_transform(X)


from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,BaggingClassifier
BC = BaggingClassifier()
RFC = RandomForestClassifier()
GBC = GradientBoostingClassifier()


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=11)

from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score,f1_score,confusion_matrix


def evaluate(X_train,X_test,y_train,y_test,model):
    pipe = make_pipeline(ct,model).fit(X_train,y_train)
    pred = pipe.predict(X_test)
    pred_prob = pipe.predict_proba(X_test)[:,1]
    acc = accuracy_score(y_test,pred)
    roc = roc_auc_score(y_test, pred_prob)
    con = confusion_matrix(y_test,pred)
    f1 = f1_score(y_test,pred)
    print('f1 score: ',f1)
    print('confusion matrix',con)
    print(f'{model.__class__.__name__}, --ACC-- {acc*100:.2f}%; --ROC-- {roc*100:.2f}%')
    return pred,pred_prob

lr_pred,lr_pred_prob = evaluate(X_train, X_test, y_train, y_test, lr)
GBC_pred,GBC_pred_prob = evaluate(X_train, X_test, y_train, y_test, GBC)
tree_pred,tree_pred_prob = evaluate(X_train, X_test, y_train, y_test, tree)
knn_pred,knn_pred_prob = evaluate(X_train, X_test, y_train, y_test, knn)
lda_pred,lda_pred_prob = evaluate(X_train, X_test, y_train, y_test, lda)
RFC_pred,RFC_pred_prob = evaluate(X_train, X_test, y_train, y_test, RFC)
BC_pred,BC_pred_prob = evaluate(X_train, X_test, y_train, y_test, BC)



def ROC(y_test,y_pred_prob,model):
    fpr,tpr, _ = roc_curve(y_test,y_pred_prob)
    plt.plot(fpr,tpr,label=model.__class__.__name__)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    

ROC(y_test, lr_pred_prob, lr)
ROC(y_test,GBC_pred_prob, GBC)
ROC(y_test,RFC_pred_prob,RFC)
ROC(y_test,tree_pred_prob,tree)
ROC(y_test,lda_pred_prob,lda)
ROC(y_test, knn_pred_prob,knn)
ROC(y_test, BC_pred_prob, BC)
plt.legend()
plt.show()




from sklearn.model_selection import cross_val_score


def cross_validation(X,y,model):
    pipe = make_pipeline(ct,model).fit(X,y)
    cv_scores = cross_val_score(pipe, X,y,cv=10,scoring='roc_auc').max()
    print(f'{model.__class__.__name__}, The cv scores using 10-fold-cross validation: {cv_scores*100:.2f}')
    return cv_scores


lr_scores = cross_validation(X,y,lr)
GBC_scores = cross_validation(X, y, GBC)
RFC_scores = cross_validation(X, y, RFC)
knn_scores = cross_validation(X, y, knn)
tree_scores = cross_validation(X, y, tree)
lda_scores = cross_validation(X, y, lda)
BC_scores = cross_validation(X, y, BC)


