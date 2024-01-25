import pandas as pd
import warnings
warnings.filterwarnings('ignore')

file_path = "C:\\ML\\python\\data\\netflix.csv"

df = pd.read_csv(file_path,delimiter=',')

df.info()
df.isna().sum()


df.columns = df.columns.str.replace(' ','_')
df.isna().sum()
print(df.corr())
df['Gender'] = df['Gender'].map({'Male':1,'Female':0})
df['Subscription_Type'].describe()
print(df.dtypes)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df1 = df.copy()


import matplotlib.pyplot as plt
import seaborn as sns


sns.heatmap(df1.corr(),annot=True)
plt.show()

def sibb(df1):
    _,axs = plt.subplots(2,2,figsize=(10,6))
    sns.countplot(x='Subscription_Type',ax=axs[0,0],data=df1)
    sns.boxplot(x='Age',y='User_ID',ax=axs[0,1],data=df1)
    sns.kdeplot(x='Monthly_Revenue',y='Age',ax=axs[1,0],data=df1)
    sns.violinplot(x='Plan_Duration',y='Gender',ax=axs[1,1],data=df1)
    plt.show()

sibb(df1)

df.nunique()
df = df.drop(['Join_Date','User_ID','Last_Payment_Date','Join_Date','Last_Payment_Date'],axis=1)

df['Subscription_Type'] = [1 if X == 'Basic' else 0 for X in df['Subscription_Type']]

df.Subscription_Type.describe()

X = df.drop('Subscription_Type',axis=1)
y = df['Subscription_Type']

from sklearn.model_selection import train_test_split


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)


from sklearn.preprocessing import OneHotEncoder,StandardScaler
ohe = OneHotEncoder(sparse_output=False)

sc = StandardScaler()


from sklearn.compose import make_column_transformer

ct = make_column_transformer(
    (ohe,X.select_dtypes(include='object').columns),
    (sc,X.select_dtypes(include=('float64')).columns),remainder='passthrough')


ct.fit_transform(X)



from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()


from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,BaggingClassifier
rfc = RandomForestClassifier()
GBC = GradientBoostingClassifier()
BC = BaggingClassifier()


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()


from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score,confusion_matrix,f1_score


def evaluate_model(model,X_train,X_test,y_train,y_test):
    pipe = make_pipeline(ct,model).fit(X_train,y_train)
    pred = pipe.predict(X_test)
    pred_prob = pipe.predict_proba(X_test)[:,1]
    acc = accuracy_score(y_test,pred)
    f1 = f1_score(y_test,pred)
    con = confusion_matrix(y_test, pred)
    roc = roc_auc_score(y_test,pred_prob)
    print(f'{model.__class__.__name__}, --Acc-- {acc*100:2f}%; --ROC-- {roc*100:.2f}; --F1-- {f1*100:.2}')
    print('Confusion Matrix',con)
    return pred,pred_prob



lr_pred,lr_pred_prob = evaluate_model(lr, X_train, X_test, y_train, y_test)
GBC_pred,GBC_pred_prob = evaluate_model(GBC, X_train, X_test, y_train, y_test)
BC_pred,BC_pred_prob = evaluate_model(BC, X_train, X_test, y_train, y_test)
rfc_pred,rfc_pred_prob = evaluate_model(rfc, X_train, X_test, y_train, y_test)
tree_pred,tree_pred_prob = evaluate_model(tree, X_train, X_test, y_train, y_test)



def ROC(y_test,y_pred_prob,model):
    fpr,tpr,_ = roc_curve(y_test,y_pred_prob)
    plt.plot(fpr,tpr,label=model.__class__.__name__)
    plt.xlabel('False Posisitve Rate')
    plt.ylabel('True Posistive Rate')
    
    
ROC(y_test,lr_pred_prob,lr)
ROC(y_test,rfc_pred_prob,rfc)
ROC(y_test, GBC_pred_prob, GBC)
plt.legend()
plt.show()

