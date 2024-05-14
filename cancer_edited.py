import pandas as pd


df = pd.read_csv('https://raw.githubusercontent.com/nickkas/python_ml_ect_/master/cancer.csv')

df.describe()
df.isna().sum()
df = df.drop('Unnamed: 32',axis=1)
print(df.dtypes)
df = df.drop('id',axis=1)

df.info()

df['diagnosis'].value_counts()

df['diagnosis'] = [1 if X == 'B' else 0 for X in df['diagnosis']]


import seaborn as sns
import matplotlib.pyplot as plt

def desc_stats(df):
    f,axs = plt.subplots(2,2,figsize=(10,6))
    sns.barplot(x='diagnosis',y='area_worst',ax=axs[0,0],data=df)
    sns.scatterplot(x='perimeter_mean',y='fractal_dimension_se',ax=axs[0,1],data=df)
    sns.scatterplot(x='texture_worst',y='texture_mean',ax=axs[1,0],data=df)
    sns.scatterplot(x='concavity_mean',y='symmetry_mean',ax=axs[1,1],data=df)
    plt.show()



def hist(df):
    his,ax1 = plt.subplots(4,3,figsize=(15,5))
    sns.countplot(df,x='diagnosis',ax=ax1[0,0])
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










X = df.drop('diagnosis',axis=1)
y = df['diagnosis']

import statsmodels.api as sm

model_ = sm.GLM(exog=sm.add_constant(X),endog=y).fit()
print(model_.summary())



from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)

from sklearn.preprocessing import MinMaxScaler

ms = MinMaxScaler()

X_train_scaled = ms.fit_transform(X_train)
X_test_scaled = ms.transform(X_test)

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,BaggingClassifier
BC = BaggingClassifier()
GBC = GradientBoostingClassifier()
RFC = RandomForestClassifier()

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score,confusion_matrix,f1_score


def eval_models(X_train_scaled,X_test_scaled,y_train,y_test,model):
    model = model.fit(X_train_scaled,y_train)
    pred = model.predict(X_test_scaled)
    pred_prob = model.predict_proba(X_test_scaled)[:,1]
    f1 = f1_score(y_test,pred)
    acc = accuracy_score(y_test, pred)
    roc = roc_auc_score(y_test,pred_prob)
    con = confusion_matrix(y_test,pred)
    print('confusion matrix',con)
    print('f1 score',f1*100)
    print(f'{model.__class__.__name__}, --ACC-- {acc*100:.2f}%; --ROC-- {roc*100:.2f}%')
    return pred,pred_prob

lr_pred,lr_pred_prob = eval_models(X_train_scaled, X_test_scaled, y_train, y_test,lr)
GBC_pred,GBC_pred_prob = eval_models(X_train_scaled, X_test_scaled, y_train, y_test, GBC)
BC_pred,BC_pred_prob = eval_models(X_train_scaled, X_test_scaled, y_train, y_test,BC)
RFC_pred,RFC_pred_prob = eval_models(X_train_scaled, X_test_scaled, y_train, y_test, RFC)


def ROC(y_test,y_pred_prob,model):
    fpr,tpr, _ = roc_curve(y_test,y_pred_prob)
    plt.plot(fpr,tpr,label=model.__class__.__name__)
    plt.title('Roc Curves of Models')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')





def main():
    desc_stats(df)
    hist(df)
    
    ROC(y_test,lr_pred_prob,lr)
    ROC(y_test,GBC_pred_prob,GBC)
    ROC(y_test,RFC_pred_prob,RFC)
    ROC(y_test,BC_pred_prob,BC)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()

