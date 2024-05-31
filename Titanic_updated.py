import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('http://bit.ly/kaggletrain')

df.info()
df.isna().sum()
print(df.dtypes)
df.nunique()

df['Survived'].plot(kind='hist')

df['Sex'] = [1 if X == 'male' else 0 for X in df['Sex']]

df['Age'].plot(kind='hist',bins=15)



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()


df1 = df.copy()


for i in df1:
    df1[i] = le.fit_transform(df1[i])



import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(df1.corr(),annot=True)
plt.show()



def desc_stats(df1):
    f,axs = plt.subplots(2,2,figsize=(10,6))
    sns.barplot(x='Embarked',y='Survived',ax=axs[0,0],data=df1)
    axs[0,0].set_xlabel('Embarked Vs Sex')
    
    sns.barplot(x='Sex',y='Survived',ax=axs[0,1],data=df1)
    axs[0,1].set_xlabel('Survived vs Sex')
    
    sns.violinplot(x=df1['Age'],ax=axs[1,0],data=df1)
    
    sns.countplot(x='Survived',ax=axs[1,1],data=df1)
    axs[1,1].set_xlabel('Number of people who survived = 1, did not survive = 0')
    plt.tight_layout()
    plt.show()







def misc(df1):
    plt_,ax1 = plt.subplots(2,3,figsize=(10,6))
    sns.kdeplot(df1[['Fare']],ax=ax1[0,0])
    sns.histplot(df1[['Survived']],ax=ax1[0,1])
    sns.violinplot(df1[['Pclass']],ax=ax1[0,2])
    sns.boxplot(df1[['SibSp']],ax=ax1[1,0])
    sns.histplot(df1[['Age']],ax=ax1[1,1])
    sns.histplot(df1[['Sex']],ax=ax1[1,2])
    plt.show()




def more_subs(copy):
    plt_2,ax2 = plt.subplots(2,2,figsize=(10,6))
    sns.barplot(df1,x='Sex',y='Survived',ax=ax2[0,0])
    sns.boxplot(df1,x='Survived',y='Age',ax=ax2[0,1])
    sns.histplot(df1[['Age']],ax=ax2[1,0])
    sns.barplot(df1,x='Sex',y='Fare',ax=ax2[1,1])
    plt.show()






X = df.drop(['PassengerId','Name','Ticket','Survived','Cabin'],axis=1)
y = df['Survived']


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)



from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)

from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='most_frequent')


from sklearn.compose import make_column_transformer

ct = make_column_transformer(
    (ohe,X.select_dtypes(include='object').columns),
    (imp,X.select_dtypes(include=['float64','int64']).columns),remainder='passthrough')


ct.fit_transform(X)


from sklearn.pipeline import make_pipeline


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,BaggingClassifier

RFC = RandomForestClassifier()
GBC = GradientBoostingClassifier()
BC = BaggingClassifier()

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()

from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score,confusion_matrix


def evaluate_titanic(X_train,X_test,y_train,y_test,model):
    pipe = make_pipeline(ct,model).fit(X_train,y_train)
    pred = pipe.predict(X_test)
    pred_prob = pipe.predict_proba(X_test)[:,1]
    acc = accuracy_score(y_test, pred)
    roc = roc_auc_score(y_test,pred_prob)
    con = confusion_matrix(y_test, pred)
    print('confusion matrix: ',con)
    print(f'{model.__class__.__name__}, --ACC-- {acc*100:.2f}%; --ROC-- {roc*100:.2f}%')
    return pred,pred_prob


lr_pred,lr_pred_prob = evaluate_titanic(X_train, X_test, y_train, y_test, lr)
GBC_pred,GBC_pred_prob = evaluate_titanic(X_train, X_test, y_train, y_test,GBC)
RFC_pred,RFC_pred_prob = evaluate_titanic(X_train, X_test, y_train, y_test, RFC)
BC_pred,BC_pred_prob = evaluate_titanic(X_train, X_test, y_train, y_test, BC)
tree_pred,tree_pred_prob = evaluate_titanic(X_train, X_test, y_train, y_test,tree)
lda_pred,lda_pred_prob = evaluate_titanic(X_train, X_test, y_train, y_test, lda)


def ROC(y_test,y_pred_prob,model):
    fpr,tpr, _ = roc_curve(y_test,y_pred_prob)
    plt.plot(fpr,tpr,label=model.__class__.__name__)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Models')



from sklearn.model_selection import cross_val_score

def cv_(X,y,model):
    pipe = make_pipeline(ct,model).fit(X,y)
    cv_scores = cross_val_score(pipe, X,y,cv=10,scoring='accuracy').max()
    print(f'{model.__class__.__name__}, --Results using 10-fold Cross Validation: {cv_scores*100:.2f}%')
    return cv_scores






def __main__():
    lr_scores=  cv_(X, y, lr)
    GBC_scors = cv_(X, y, GBC)
    RFC_scores = cv_(X, y,RFC)
    lda_scores = cv_(X, y, lda)
    tree_scores = cv_(X, y, tree)
    BC_scores = cv_(X, y, BC)
    desc_stats(df1)
    misc(df1)
    more_subs(df1)
    ROC(y_test, lr_pred_prob, lr)
    ROC(y_test,RFC_pred_prob,RFC)
    ROC(y_test,GBC_pred_prob,GBC)
    ROC(y_test,lda_pred_prob,lda)
    ROC(y_test,BC_pred_prob,BC)
    ROC(y_test,tree_pred_prob,tree)
    plt.legend()
    plt.show()


if "__name__" == __main__():
    __main__





