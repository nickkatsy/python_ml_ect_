import pandas as pd
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('https://raw.githubusercontent.com/JoshuaEubanksUCF/Black_and_Gold_Analytics/main/Student_Basket_Data/BGA_basketball_data.csv')

df.info()
df.isna().sum()


print(df['Made'].describe())

df['Major'].value_counts()
df['Major'] = df['Major'].str.replace(',', '-', regex=True) 


df.isna().sum()

import matplotlib.pyplot as plt
import seaborn as sns


sns.heatmap(df.corr(), annot=True)


def desc_subplots(df):
    plt_,axs = plt.subplots(2,2,figsize=(12,6))
    sns.barplot(df,x='Height',y='Made',ax=axs[0,0])
    sns.barplot(df,x='Distance',y='Made',ax=axs[0,1])
    sns.barplot(df,x='Major',y='Made',ax=axs[1,0])
    sns.countplot(df,x='Made',ax=axs[1,1])
    plt.show()


desc_subplots(df)



df['Major'] = pd.factorize(df['Major'])[0]
df['Name'] = pd.factorize(df['Name'])[0]



X = df.drop('Made',axis=1)
y = df[['Made']]




from sklearn.model_selection import train_test_split


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=0)


from sklearn.linear_model import LogisticRegression

clf = LogisticRegression().fit(X_train,y_train)
clf_pred = clf.predict(X_test)
clf_pred_prob = clf.predict_proba(X_test)[::,1]


from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

rfc = RandomForestClassifier().fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)
rfc_pred_prob = rfc.predict_proba(X_test)[::,1]



gbc = GradientBoostingClassifier().fit(X_train,y_train)
gbc_pred = gbc.predict(X_test)
gbc_pred_prob = gbc.predict_proba(X_test)[::,1]

from sklearn.naive_bayes import GaussianNB

nbc = GaussianNB().fit(X_train,y_train)
nbc_pred = nbc.predict(X_test)
nbc_pred_prob = nbc.predict_proba(X_test)[::,1]





# Results from each model
from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score


def evaluate_model(model_name,y_true,y_pred,y_pred_prob):
    acc = accuracy_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_pred_prob)
    print(f'{model_name} - Accuracy: {acc * 100:.2f}%, ROC-AUC: {roc * 100:.2f}%')

evaluate_model('Logistic Regression', y_test,clf_pred,clf_pred_prob)
evaluate_model('Random Forest', y_test,rfc_pred,rfc_pred_prob)
evaluate_model('Naive Bayes', y_test,nbc_pred,nbc_pred_prob)
evaluate_model('Gradient Boosting',y_test,gbc_pred,gbc_pred_prob)






# ROC curves plotted based on the results of the models

def roc_curve_plot(y_test, y_pred_prob, model_name):
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.plot(fpr,tpr,label=model_name)
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

roc_curve_plot(y_test,clf_pred_prob,'Logistic Regression')
roc_curve_plot(y_test,nbc_pred_prob,'Naive Bayes')
roc_curve_plot(y_test,gbc_pred_prob,'Gradient Boosting')
roc_curve_plot(y_test,rfc_pred_prob,'Random Forest')
plt.legend()
plt.show()
