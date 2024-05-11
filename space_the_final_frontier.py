import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('https://raw.githubusercontent.com/nickkatsy/python_ml_ect_/master/space_train.csv')

df.isna().sum()
df.nunique()

df.info()
df['Transported'] = df['Transported'].map({True:1,False:0})
print(df.dtypes)
df['HomePlanet'].value_counts()

df.dtypes
df['Name'].describe()
df['PassengerId'].describe()


df = df.drop(['Cabin','PassengerId','Name'],axis=1)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df1 = df.copy()

for i in df1:
    df1[i] = le.fit_transform(df1[i])
    
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(df1.corr(), annot=True)
plt.show()



df['HomePlanet'].value_counts()

def stuff(df1):
    f,axs = plt.subplots(2,2,figsize=(10,6))
    sns.barplot(x='HomePlanet',y='Destination',ax=axs[0,0],data=df1)
    sns.boxplot(x='VIP',y='Destination',ax=axs[0,1],data=df1)
    sns.barplot(x='CryoSleep',y='VIP',ax=axs[1,0],data=df1)
    sns.barplot(x='VIP',y='Transported',ax=axs[1,1],data=df1)
    plt.show()
    
    
stuff(df1)




X = df.drop(['Transported'],axis=1)
y = df['Transported']

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)



from sklearn.impute import SimpleImputer

imp = SimpleImputer()

from sklearn.compose import make_column_transformer

ct = make_column_transformer(
    (ohe,X.select_dtypes(include='object').columns),
    (imp,X.select_dtypes(include=['int64','float64']).columns),remainder='passthrough')



ct.fit_transform(X)


from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,BaggingClassifier
BC = BaggingClassifier()
GBC = GradientBoostingClassifier()
RFC = RandomForestClassifier()


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()

from sklearn.tree import DecisionTreeClassifier
Tree = DecisionTreeClassifier()


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)


from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score,confusion_matrix,f1_score

def ecv(X_train,X_test,y_train,y_test,model):
    pipe = make_pipeline(ct,model).fit(X_train,y_train)
    pred = pipe.predict(X_test)
    pred_prob = pipe.predict_proba(X_test)[:,1]
    acc = accuracy_score(y_test, pred)
    roc = roc_auc_score(y_test, pred_prob)
    con = confusion_matrix(y_test, pred)
    f1 = f1_score(y_test,pred)
    print('confusion matrix',con)
    print('f1 score',round(f1*100))
    print(f'{model.__class__.__name__}, --ACC-- {acc*100:.2f}%; -ROC- {roc*100:.2f}%')
    return pred,pred_prob


lr_pred,lr_pred_prob = ecv(X_train, X_test, y_train, y_test,lr)
GBC_pred,GBC_pred_prob = ecv(X_train, X_test, y_train, y_test, GBC)
RFC_pred,RFC_pred_prob = ecv(X_train, X_test, y_train, y_test, RFC)
knn_pred,knn_pred_prob = ecv(X_train, X_test, y_train, y_test, knn)
BC_pred,BC_pred_prob = ecv(X_train, X_test, y_train, y_test, BC)
lda_pred,lda_pred_prob = ecv(X_train, X_test, y_train, y_test, lda)
Tree_pred,Tree_pred_prob = ecv(X_train, X_test, y_train, y_test, Tree)

def ROC(y_test,y_pred_prob,model):
    fpr,tpr, _ = roc_curve(y_test,y_pred_prob)
    plt.plot(fpr,tpr,label=model.__class__.__name__)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')


ROC(y_test,BC_pred_prob,BC)
ROC(y_test,GBC_pred_prob,GBC)
ROC(y_test,RFC_pred_prob,RFC)
ROC(y_test,lr_pred_prob,lr)
ROC(y_test,Tree_pred_prob,Tree)
ROC(y_test,lda_pred_prob,lda)
ROC(y_test,knn_pred_prob,knn)
plt.legend()
plt.show()


