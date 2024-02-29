import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('https://raw.githubusercontent.com/nickkas/python_ml_ect_/master/cancer.csv')


df.info()

df.isna().sum()

df.nunique()

print(df.dtypes)
df.shape

df = df.drop(['Unnamed: 32','id'],axis=1)

df.corr()
df['diagnosis'].value_counts()


df['diagnosis'] = pd.get_dummies(df.diagnosis,prefix='diagnosis').iloc[:,0:1]

def subplots(df):
    his,ax1 = plt.subplots(4,3,figsize=(12,6))
    sns.kdeplot(df,x='diagnosis',ax=ax1[0,0])
    sns.histplot(df,x='radius_mean',ax=ax1[0,1])
    sns.histplot(df,x='texture_mean',ax=ax1[0,2])
    sns.histplot(df,x='perimeter_mean',ax=ax1[1,0])
    sns.histplot(df,x='area_mean',ax=ax1[1,1])
    sns.histplot(df,x='smoothness_mean',ax=ax1[1,2])
    sns.histplot(df,x='smoothness_mean',ax=ax1[2,0])
    sns.histplot(df,x='concavity_mean',ax=ax1[2,1])
    sns.histplot(df,x='compactness_mean',ax=ax1[2,2])   
    sns.histplot(df,x='concave points_mean',ax=ax1[3,0])
    sns.histplot(df,x='symmetry_mean',ax=ax1[3,1])
    sns.histplot(df,x='fractal_dimension_mean',ax=ax1[3,2])
    plt.show()



subplots(df)







X = df.drop('diagnosis',axis=1)
y = df[['diagnosis']]

y.value_counts(normalize=True)





from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)



from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,BaggingClassifier
from sklearn.naive_bayes import GaussianNB


lr = LogisticRegression()

rfc = RandomForestClassifier()
gbr = GradientBoostingClassifier()
NB = GaussianNB()
BC = BaggingClassifier()








from sklearn.metrics import roc_auc_score,accuracy_score,confusion_matrix,f1_score



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
NB_pred,NB_pred_prob = evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test, NB)
BC_pred,BC_pred_prob = evaluate_model(X_train, X_test, y_train, y_test,BC)
gbr_pred,gbr_pred_prob = evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test,gbr)
    
