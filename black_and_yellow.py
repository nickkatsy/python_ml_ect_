import pandas as pd
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('https://raw.githubusercontent.com/JoshuaEubanksUCF/Black_and_Gold_Analytics/main/Student_Basket_Data/BGA_basketball_data.csv')

df.info()
df.corr()


print(df['Made'].describe())

df['Major'].value_counts()
df['Major'] = df['Major'].str.replace(',', '-', regex=True) 


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



df['Major'] = pd.factorize(df['Major'])[0]
df['Name'] = pd.factorize(df['Name'])[0]



X = df.drop('Made',axis=1)
y = df[['Made']]

y.value_counts(normalize=True)

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


from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier().fit(X,y)
gbc_pred = gbc.predict(X)
gbc_pred_prob = gbc.predict_proba(X)[::,1]

from sklearn.naive_bayes import GaussianNB

nbc = GaussianNB().fit(X,y)
nbc_pred = nbc.predict(X)
nbc_pred_prob = nbc.predict_proba(X)[::,1]






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

#Scoring using Gradient Boost classifier
grad_acc = accuracy_score(y, gbc_pred)
print(f'the accuracy using Gradinet boost classifier: {grad_acc}')

roc_grad = roc_auc_score(y, gbc_pred_prob)
print(f'the roc_auc score using Gradient boost: {roc_grad}')



#scoring using Naive Bayes

nb_accuracy = accuracy_score(y, nbc_pred)
print(f'the accuracy using Naive Bayes: {nb_accuracy}')

nb_roc = roc_auc_score(y, nbc_pred_prob)
print(f'the roc_auc score using Naive Bayes: {nb_roc}')





# ROC curves plotted based on the results of the models

def roc_logistic_regression(y, y_pred_prob):
    fpr, tpr, _ = roc_curve(y, y_pred_prob)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve From Logistic Regression')
    plt.show()

roc_logistic_regression(y, clf_pred_prob)


def roc_forrest(y,rfc_pred_prob):
    fpr,tpr, _ = roc_curve(y, rfc_pred_prob)
    plt.plot(fpr,tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve from Random Forest')
    plt.show()


roc_forrest(y, rfc_pred_prob)



def roc_gradient_boost(y,gbc_pred_prob):
    fpr,tpr, _ = roc_curve(y, gbc_pred_prob)
    plt.plot(fpr,tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve from Gradient Boost')
    plt.show()


roc_gradient_boost(y, gbc_pred_prob)


def roc_naive_bayes(y,nbc_pred_prob):
    fpr,tpr, _ = roc_curve(y, nbc_pred_prob)
    plt.plot(fpr,tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve from Naive Bayes')
    plt.show()



roc_naive_bayes(y, nbc_pred_prob)
