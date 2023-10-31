import pandas as pd
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('https://raw.githubusercontent.com/JoshuaEubanksUCF/Black_and_Gold_Analytics/main/Student_Basket_Data/BGA_basketball_data.csv')

df.info()
df.corr()


print(df['Made'].describe())

df['Major'].value_counts()
df['Major'] = df['Major'].str.replace(',', '-', regex=True) 

df_cv = df.copy


df.isna().sum()

import matplotlib.pyplot as plt
import seaborn as sns


sns.heatmap(df.corr(), annot=True)


def desc_subplots(df):
    plt_,axs = plt.subplots(2,2,figsize=(10,6))
    sns.barplot(df,x='Height',y='Major',ax=axs[0,0])
    sns.scatterplot(df,x='Distance',y='Major',ax=axs[0,1])
    sns.barplot(df,x='Made',y='Major',ax=axs[1,0])
    sns.countplot(df,x='Major',ax=axs[1,1])
    plt.show()


desc_subplots(df)

#using labelencoder for the sake of time

df1 = df.copy()


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for i in df1:
    df1[i] = le.fit_transform(df1[i])


import statsmodels.api as sm


model_1 = sm.GLM(exog=df1['Major'],endog=df1['Made']).fit()
print(model_1.summary())


model_2 = sm.GLM(exog=df1['Height'], endog=df1['Made']).fit()
print(model_2.summary())


model_3 = sm.GLM(exog=df1['Distance'],endog=df1['Made']).fit()
print(model_3.summary())


model_4 = sm.GLM(exog=df1['Name'],endog=df1['Made']).fit()
print(model_4.summary())



df = df.drop('Name',axis=1)



df['Major'] = pd.factorize(df['Major'])[0]


df_cv = df.copy()

X = df.drop('Made',axis=1)
y = df[['Made']]


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit_transform(X)


from sklearn.linear_model import LogisticRegression

clf = LogisticRegression().fit(X,y)
clf_pred = clf.predict(X)
clf_pred_prob = clf.predict_proba(X)[::,1]


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier().fit(X,y)
rfc_pred = rfc.predict(X)
rfc_pred_prob = rfc.predict_proba(X)[::,1]


from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score


#Results from Logistic Regression

acc_clf = accuracy_score(y, clf_pred)
print(f'The accuracy for the logistic regression model is: {acc_clf*100}')

roc_clf = roc_auc_score(y, clf_pred_prob)
print(f'the roc score for the logistic regression model: {roc_clf*100}')


#scores using random forrest

acc_rfc = accuracy_score(y, rfc_pred)
print(f'the accuaracy using random forrest is: {acc_rfc*100}')


rfc_roc = roc_auc_score(y,rfc_pred_prob)
print(f'the roc score using random forest: {rfc_roc*100}')



def roc_logistic_regression(y_true, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()

roc_logistic_regression(y, clf_pred_prob)
    
    


def roc_Random_Forest(y_true, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_true, rfc_pred_prob)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()

roc_Random_Forest(y, rfc_pred_prob)








