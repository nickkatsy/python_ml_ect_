import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('https://raw.githubusercontent.com/LeeMorinUCF/QMB6358S22/main/demo_24_Classification/credit.csv')


df.info()

df.describe()
df.isna().sum()

df.nunique()

import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(df.corr(), annot=True)



def scatter(dataframe):
    plt_,axs = plt.subplots(2,2,figsize=(10,6))
    sns.scatterplot(df,x='age',y='amount',ax=axs[0,0])
    sns.scatterplot(df,x='purpose',y='amount',ax=axs[0,1])
    sns.scatterplot(df,x='duration',y='amount',ax=axs[1,0])
    sns.scatterplot(df,x='age',y='duration',ax=axs[1,1])
    plt.show()
    
    
scatter(df)


def count(dataframe):
    plt__,axs2 = plt.subplots(2,2,figsize=(10,6))
    sns.boxplot(df[['amount']],ax=axs2[0,0])
    sns.boxplot(df[['installment']],ax=axs2[0,1])
    sns.violinplot(df[['Default']],ax=axs2[1,0])
    sns.boxplot(df[['cards']],ax=axs2[1,1])
    
    plt.show()


count(df)


def bar(dataframe):
    plt_,axs3 = plt.subplots(2,2,figsize=(10,6))
    sns.barplot(df,x='Default',y='age',ax=axs3[0,0])
    sns.barplot(df,x='Default',y='duration',ax=axs3[0,1])
    sns.barplot(df,x='Default',y='installment',ax=axs3[1,0])
    sns.kdeplot(df,x='Default',ax=axs3[1,1])
    plt.show()
    


bar(df)

X = df.drop('Default',axis=1)
y = df[['Default']]

from sklearn.model_selection import train_test_split


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)


from sklearn.linear_model import LogisticRegression



clf = LogisticRegression()


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()


from sklearn.tree import DecisionTreeClassifier

trees = DecisionTreeClassifier()



from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer


ohe = OneHotEncoder()


ct = make_column_transformer(
    (ohe,['employ','checkingstatus1','history','purpose','savings','status','others','property','otherplans','housing','job','tele','foreign']),remainder='passthrough')



ct.fit_transform(X)


from sklearn.pipeline import make_pipeline

# pipeline for logistic regression
pipe_clf = make_pipeline(ct,clf).fit(X_train,y_train)

pipe_pred = pipe_clf.predict(X_test)

pipe_pred_prob = pipe_clf.predict_proba(X_test)[::,1]



from sklearn.metrics import accuracy_score,f1_score,roc_curve,roc_auc_score



F1_score_clf = f1_score(y_test, pipe_pred)
print(F1_score_clf)



acc = accuracy_score(y_test, pipe_pred)
print('Logistic Regression Score: ',acc*100)

roc_clf = roc_auc_score(y_test, pipe_pred_prob)
print('ROC score for',roc_clf*100)





fpr, tpr, _ = roc_curve(y_test, pipe_pred_prob)
plt.plot(fpr,tpr)
plt.title('ROC Curve For Logistic Regression')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()












