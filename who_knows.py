import pandas as pd
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('https://raw.githubusercontent.com/nickkatsy/python_ml_ect_/master/data.csv',sep=';')
df.info()
df.columns = df.columns.str.replace(' ','_')
print(df.dtypes)
df['time_of_day'] = df['Daytime/evening_attendance\t']

df = df.drop('Daytime/evening_attendance\t',axis=1)

df.nunique()
df['Admission_grade'].describe()
df.isna().sum()
df.describe()


df['Target'].value_counts()





df['Target'] = df['Target'].apply(lambda X: 1 if X == 'Enrolled' else 0)



import seaborn as sns
import matplotlib.pyplot as plt

f = plt.figure(figsize=(15,10))
sns.heatmap(df.corr(), annot=False)
plt.show()


def plotss(df):
    _,axs = plt.subplots(3,3,figsize=(10,6))
    sns.barplot(x='Course',y='Application_mode',ax=axs[0,0],data=df)
    sns.boxplot(x='Target',y='Admission_grade',ax=axs[0,1],data=df)
    sns.lineplot(x='Target',y='Previous_qualification_(grade)',ax=axs[0,2],data=df)
    sns.boxplot(x='Course',y='Target',ax=axs[1,0],data=df)
    sns.boxplot(x='Curricular_units_2nd_sem_(enrolled)',y='Curricular_units_2nd_sem_(grade)',ax=axs[1,2],data=df)
    sns.violinplot(x='Target',y='Age_at_enrollment',ax=axs[1,1],data=df)
    sns.barplot(x='Target',y='Application_order',ax=axs[1,2],data=df)
    sns.kdeplot(x='Curricular_units_1st_sem_(grade)',y='Curricular_units_2nd_sem_(approved)',ax=axs[2,0],data=df)
    sns.violinplot(x='time_of_day',y='Application_mode',ax=axs[2,0],data=df)
    sns.countplot(x='Target',ax=axs[2,1],data=df)
    sns.boxplot(x='Curricular_units_2nd_sem_(without_evaluations)',y='Previous_qualification_(grade)',ax=axs[2,2],data=df)
    plt.show()
    
    
    
plotss(df)



X = df.drop('Target',axis=1)
y = df['Target']



from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)

from sklearn.preprocessing import OneHotEncoder,StandardScaler


ohe = OneHotEncoder()

sc = StandardScaler()

from sklearn.compose import make_column_transformer



ct = make_column_transformer(
    (ohe, X.select_dtypes(include='object').columns),
    (sc, X.select_dtypes(['int64','float64']).columns),
    remainder='passthrough'
)


ct.fit_transform(X)


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


from sklearn.pipeline import make_pipeline


from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score,confusion_matrix


def evaluate(model,X_train,X_test,y_train,y_test):
    pipe = make_pipeline(ct,model).fit(X_train,y_train)
    pred = pipe.predict(X_test)
    pred_prob = pipe.predict_proba(X_test)[:,1]
    acc = accuracy_score(y_test,pred)
    con = confusion_matrix(y_test, pred)
    roc = roc_auc_score(y_test, pred_prob)
    print('Confusion Matrix: ',con)
    print(f'{model.__class__.__name__}, --Accuracy-- {acc*100:.2f}%; --roc-- {roc*100:.2f}%')
    return pred,pred_prob



lr_pred,lr_pred_prob = evaluate(lr, X_train, X_test, y_train, y_test)
gbc_pred,gbc_pred_prob = evaluate(gbc, X_train, X_test, y_train, y_test)
BC_pred,BC_pred_prob = evaluate(BC, X_train, X_test, y_train, y_test)
tree_pred,tree_pred_prob = evaluate(tree, X_train, X_test, y_train, y_test)
nb_pred,nb_pred_prob = evaluate(nb, X_train, X_test, y_train, y_test)
rfc_pred,rfc_pred_prob = evaluate(rfc, X_train, X_test, y_train, y_test)




def ROC_CURVES(y_test,y_pred_prob,model):
    fpr,tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.plot(fpr,tpr,label=model.__class__.__name__)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    
    
    
ROC_CURVES(y_test, lr_pred_prob, lr)
ROC_CURVES(y_test, rfc_pred_prob, rfc)
ROC_CURVES(y_test, gbc_pred_prob, gbc)
ROC_CURVES(y_test, BC_pred_prob, BC)
ROC_CURVES(y_test, tree_pred_prob, tree)
plt.legend()
plt.show()
