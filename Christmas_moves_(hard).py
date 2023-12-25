import pandas as pd
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('C:/ml/python/data/christmas_movies.csv',delimiter=',')


df.info()
df.isna().sum()

obj_cols = df.select_dtypes(include='object').columns
for col in obj_cols:
    print(f'Unique values in {col}: {df[col].unique()}')

df.nunique()
df['type'].nunique()
df['type'] = [1 if X == 'Movie' else 0 for X in df['type']]




X = df.drop('type',axis=1)
y = df['type']


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)
X.isna().sum()

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False,handle_unknown='ignore')

from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='most_frequent')

from sklearn.compose import make_column_transformer

ct = make_column_transformer(
    (ohe, X.select_dtypes(include='object').columns),
    (imp,X.select_dtypes(include=['int64','float64']).columns),
    remainder='passthrough'
)


ct.fit_transform(X)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
rfc = RandomForestClassifier()
gbc = GradientBoostingClassifier()

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()

from sklearn.pipeline import make_pipeline

clf_pipe = make_pipeline(ct,clf).fit(X_train,y_train)
clf_pred = clf_pipe.predict(X_test)
clf_pred_prob = clf_pipe.predict_proba(X_test)[::,1]


lda_pipe = make_pipeline(ct,lda).fit(X_train,y_train)
lda_pred = lda_pipe.predict(X_test)
lda_pred_prob = lda_pipe.predict_proba(X_test)[::,1]

nb_pipe = make_pipeline(ct,nb).fit(X_train,y_train)
nb_pred = nb_pipe.predict(X_test)
nb_pred_prob = nb_pipe.predict_proba(X_test)[::,1]

gbc_pipe = make_pipeline(ct,gbc).fit(X_train,y_train)
gbc_pred = gbc_pipe.predict(X_test)
gbc_pred_prob = gbc_pipe.predict_proba(X_test)[::,1]

rfc_pipe = make_pipeline(ct,rfc).fit(X_train,y_train)
rfc_pred = rfc_pipe.predict(X_test)
rfc_pred_prob = rfc_pipe.predict_proba(X_test)[::,1]


from sklearn.metrics import roc_auc_score,accuracy_score,confusion_matrix,roc_curve

def evaluate_model(y_test,y_pred,y_pred_prob,model_name):
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred_prob)
    print(f'{model_name} --Accuracy-- {acc*100:.2f}%, --ROC-- {roc*100:.2f}%')
    con = confusion_matrix(y_test, y_pred)
    print(con)
    
    
evaluate_model(y_test, clf_pred, clf_pred_prob, 'Logistic Regression')
evaluate_model(y_test, rfc_pred, rfc_pred_prob, 'Random Forest')
evaluate_model(y_test, gbc_pred, gbc_pred_prob, 'Gradient Boost')
evaluate_model(y_test, nb_pred, nb_pred_prob, 'Naive Bayes')
evaluate_model(y_test, lda_pred, lda_pred_prob, 'LDA')


import matplotlib.pyplot as plt


def ROC(y_test,y_pred_prob,model_name):
    fpr,tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.plot(fpr,tpr,label=model_name)
    plt.xlabel('False Positve Rate')
    plt.ylabel('True Positive Rate')
    


ROC(y_test,clf_pred_prob,'Logistic Regression')
ROC(y_test,rfc_pred_prob,'Random Forest')
ROC(y_test,gbc_pred_prob,'Gradient Boost')
ROC(y_test,nb_pred_prob,'Naive Bayes')
ROC(y_test,lda_pred_prob,'LDA')
plt.legend()
plt.show()


