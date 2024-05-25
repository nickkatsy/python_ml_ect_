import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('C:/ML/python/data/train.csv',delimiter=',')

df.info()

df = df.drop('ID',axis=1)

df['Gender'] = [1 if X == 'Male' else 0 for X in df['Gender']]

df['Graduated'] = [1 if X == 'Yes' else 0 for X in df['Graduated']]
df['Ever_Married'] = [1 if X == 'Yes' else 0 for X in df['Ever_Married']]
df.nunique()
df.dtypes
df['Segmentation'].value_counts()


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df1 = df.copy()

for i in df1:
    df1[i] = le.fit_transform(df1[i])


import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(df1.corr(), annot=True)
plt.show()




def subs(df1):
    f,axs = plt.subplots(2,2,figsize=(10,6))
    sns.boxplot(x='Segmentation',y='Profession',ax=axs[0,0],data=df1)
    sns.boxplot(x='Graduated',y='Segmentation',ax=axs[0,1],data=df1)
    sns.barplot(x='Segmentation',y='Ever_Married',ax=axs[1,0],data=df1)
    sns.lineplot(x='Age',y='Segmentation',ax=axs[1,1],data=df1)
    plt.show()
    
    
subs(df1)


X = df.drop('Graduated',axis=1)
y = df['Graduated']

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
from sklearn.impute import SimpleImputer
imp = SimpleImputer()

from sklearn.compose import make_column_transformer

ct = make_column_transformer(
    (ohe,X.select_dtypes(include='object').columns),
    (imp,X.select_dtypes(include=['float64','int64']).columns),remainder='passthrough')


ct.fit_transform(X)


from sklearn.pipeline import make_pipeline


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,BaggingClassifier
BC = BaggingClassifier()
GBC = GradientBoostingClassifier()
RFC = RandomForestClassifier()

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()






from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score


def evaluate(X_train,X_test,y_train,y_test,model):
    pipe = make_pipeline(ct,model).fit(X_train,y_train)
    pred = pipe.predict(X_test)
    pred_prob = pipe.predict_proba(X_test)[:,1]
    acc = accuracy_score(y_test,pred)
    roc = roc_auc_score(y_test, pred_prob)
    print(f'{model.__class__.__name__}, --ACC-- {acc*100:.2f}%; --ROC-- {roc*100:.2f}%')
    return pred,pred_prob

lr_pred,lr_pred_prob = evaluate(X_train, X_test, y_train, y_test,lr)
GBC_pred,GBC_pred_prob = evaluate(X_train, X_test, y_train, y_test, GBC)
RFC_pred,RFC_pred_prob = evaluate(X_train, X_test, y_train, y_test, RFC)
tree_pred,tree_pred_prob = evaluate(X_train, X_test, y_train, y_test, tree)
lda_pred,lda_pred_prob = evaluate(X_train, X_test, y_train, y_test, lda)







def ROC(y_test,y_pred_prob,model):
    fpr,tpr, _ = roc_curve(y_test,y_pred_prob)
    plt.plot(fpr,tpr,label=model.__class__.__name__)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positve Rate')
    plt.title('ROC Curves')


ROC(y_test,lr_pred_prob,lr)
ROC(y_test,GBC_pred_prob,GBC)
ROC(y_test,RFC_pred_prob,RFC)
ROC(y_test,tree_pred_prob,tree)
plt.legend()
plt.show()

    




from sklearn.model_selection import cross_val_score

def cv_val(X,y,model):
    pipe = make_pipeline(ct,model).fit(X,y)
    cv_scores = cross_val_score(pipe, X,y,cv=10,scoring='roc_auc').max()
    print(f'{model.__class__.__name__}, --Cross Validation Scores-- {cv_scores*100:.2f}%')
    return cv_scores


lr_scores = cv_val(X, y, lr)
GBC_scores = cv_val(X,y,GBC)
tree_scores = cv_val(X, y, tree)
RFC_scores = cv_val(X, y,RFC)
lda_scores = cv_val(X, y, lda)




