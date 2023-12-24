import pandas as pd
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('C:/ml/python/data/income.csv',delimiter=',')

df.isna().sum()
df.describe()
df.info()

df['income'].describe()

df['income'] = [1 if X == '>50K' else 0 for X in df['income']]

df.columns = df.columns.str.replace('.','_')

df1 = df.copy()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

obj_col = df.select_dtypes('object').columns

for i in df1:
    df1[i] = le.fit_transform(df1[i])

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
sns.heatmap(df1.corr(), annot=True)


def subplots(df1):
    plt_, axs = plt.subplots(2, 2, figsize=(12,10))

    sns.barplot(x='education', y='income',data=df1,ax=axs[0,0])
    axs[0,0].set_title('Education vs. Income')

    sns.kdeplot(x='age',y='hours_per_week',hue='education',data=df1,fill=True,ax=axs[0,1])
    axs[0,1].set_title('Age vs. Hours per Week')

    sns.countplot(x='marital_status', hue='income',data=df1,ax=axs[1,0])
    axs[1,0].set_title('Marital Status')

    sns.boxplot(x='income',y='age',data=df1,ax=axs[1,1])
    axs[1,1].set_title('Age vs. Income')

    plt.show()

subplots(df1)

import statsmodels.api as sm

model_ = sm.OLS(exog=df1.drop('income',axis=1),endog=(df1['income'])).fit()
print(model_.summary())


X = df.drop('income',axis=1)
y = df['income']


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

from sklearn.compose import make_column_transformer


ct = make_column_transformer(
    (ohe,X.select_dtypes(include='object').columns),
    (sc,X.select_dtypes(include=['int64','float64']).columns),remainder='passthrough')

ct.fit_transform(X)





from sklearn.linear_model import LogisticRegression
clf_lr = LogisticRegression()


from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

rfc = RandomForestClassifier()

gbc = GradientBoostingClassifier()

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=7)


from sklearn.pipeline import make_pipeline

lr_pipe = make_pipeline(ct,clf_lr).fit(X_train,y_train)
lr_pred = lr_pipe.predict(X_test)
lr_pred_prob = lr_pipe.predict_proba(X_test)[::,1]


rfc_pipe = make_pipeline(ct,rfc).fit(X_train,y_train)
rfc_pred = rfc_pipe.predict(X_test)
rfc_pred_prob = rfc_pipe.predict_proba(X_test)[::,1]


gbc_pipe = make_pipeline(ct,gbc).fit(X_train,y_train)
gbc_pred = gbc_pipe.predict(X_test)
gbc_pred_prob = gbc_pipe.predict_proba(X_test)[::,1]


nb_pipe = make_pipeline(ct,nb).fit(X_train,y_train)
nb_pred = nb_pipe.predict(X_test)
nb_pred_prob = nb_pipe.predict_proba(X_test)[::,1]

knn_pipe = make_pipeline(ct,knn).fit(X_train,y_train)
knn_pred = knn_pipe.predict(X_test)
knn_pred_prob = knn_pipe.predict_proba(X_test)[::,1]

tree_pipe = make_pipeline(ct,tree).fit(X_train,y_train)
tree_pred = tree_pipe.predict(X_test)
tree_pred_prob = tree_pipe.predict_proba(X_test)[::,1]

lda_pipe = make_pipeline(ct,lda).fit(X_train,y_train)
lda_pred = lda_pipe.predict(X_test)
lda_pred_prob = lda_pipe.predict_proba(X_test)[::,1]


from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score,f1_score,confusion_matrix

def evaluate_model(y_test,y_pred,y_pred_prob,model_name):
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred_prob)
    cm = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f'{model_name} -Accuracy- {acc*100:.2f}%, -ROC_AUC- {roc*100:.2f}%, -F1 Score- {f1*100:.2f}%')
    print('Confusion Matrix:')
    print(cm)
    



evaluate_model(y_test, lr_pred, lr_pred_prob,'Logistic Regression')
evaluate_model(y_test, rfc_pred, rfc_pred_prob, 'Random Forest')
evaluate_model(y_test, tree_pred, tree_pred_prob, 'Decision Tree')
evaluate_model(y_test, lda_pred, lda_pred_prob, 'LDA')
evaluate_model(y_test, knn_pred, knn_pred_prob, 'K-Nearest Neighbors')
evaluate_model(y_test, nb_pred, nb_pred_prob, 'Naive Bayes')
evaluate_model(y_test, gbc_pred, gbc_pred_prob, 'Gradient Boost')



def ROC(y_test,y_pred_prob,model_name):
    fpr,tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.plot(fpr,tpr,label=model_name)
    plt.xlabel('True Positive Rate')
    plt.ylabel('False Positive Rate')
    


ROC(y_test, lr_pred_prob, 'Logistic Regression')
ROC(y_test, gbc_pred_prob, 'Gradient Boost')
ROC(y_test,nb_pred_prob,'Naive Bayes')
ROC(y_test,tree_pred_prob,'Decision Tree')
ROC(y_test,lda_pred_prob,'LDA')
ROC(y_test, knn_pred_prob, 'K-Nearest Neighbors')
ROC(y_test, rfc_pred_prob, 'Random Forest')
plt.legend()
plt.show()
