import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('https://raw.githubusercontent.com/LeeMorinUCF/QMB6358F23/main/demo_14_linear_models_in_python/housing_data.csv')


df.info()

df.isna().sum()

df.corr()

df.nunique()


df = df.drop('obsn_num',axis=1)


import seaborn as sns
import matplotlib.pyplot as plt

def basic_subplots(df):
  _,axs = plt.subplots(2,2,figsize=(12,6))
  sns.scatterplot(df,x='income',y='house_price',ax=axs[0,0])
  sns.kdeplot(df,x='in_cali',ax=axs[0,1])
  sns.violinplot(df['income'],ax=axs[1,0])
  sns.distplot(df['earthquake'],ax=axs[1,1])
  plt.show()


basic_subplots(df)

# Selection

X = df.drop('in_cali',axis=1)
y = df[['in_cali']]


from sklearn.model_selection import train_test_split


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)


# scaling the features

from sklearn.preprocessing import StandardScaler


sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# Creating Classifier
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()


from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier,BaggingClassifier

gbc = GradientBoostingClassifier()
rfc = RandomForestClassifier()
BC = BaggingClassifier()








# ROC_AUC and Accuracy Scores
from sklearn.metrics import accuracy_score,roc_auc_score


def evaluate_model(model,X_train_scaled,X_test_scaled,y_train,y_test):
    model = model.fit(X_train_scaled,y_train)
    pred = model.predict(X_test_scaled)
    pred_prob = model.predict_proba(X_test_scaled)[::,1]
    acc = accuracy_score(y_test, pred)
    roc = roc_auc_score(y_test, pred_prob)
    print(f'{model.__class__.__name__}, --Accuracy-- {acc*100:.2f}%; --ROC-- {roc*100:.2f}%')
    return pred,pred_prob


lr_pred,lr_pred_prob = evaluate_model(lr, X_train_scaled, X_test_scaled, y_train, y_test)
nb_pred,nb_pred_prob = evaluate_model(nb, X_train_scaled, X_test_scaled, y_train, y_test)
gbc_pred,gbc_pred_prob = evaluate_model(gbc, X_train_scaled, X_test_scaled, y_train, y_test)
rfc_pred,rfc_pred_prob = evaluate_model(rfc, X_train_scaled, X_test_scaled, y_train, y_test)
BC_pred,BC_pred_prob = evaluate_model(BC, X_train_scaled, X_test_scaled, y_train, y_test)


# cross-val scores
from sklearn.model_selection import cross_val_score


def cross_validation(model,X_train_scaled,y_train):
    cv_scores = cross_val_score(model, X_train_scaled,y_train, cv=5,scoring='roc_auc').mean()
    print(f'{model.__class__.__name__}, --ROC_AUC Score 5-fold Cross Valiation-- {cv_scores*100:.2f}%')
    return cv_scores


lr_scores = cross_validation(lr, X_train_scaled, y_train)
nb_scores = cross_validation(nb, X_train_scaled, y_train)
gbc_scores = cross_validation(gbc, X_train_scaled, y_train)
rfc_scores = cross_validation(rfc, X_train_scaled, y_train)
BC_scores = cross_validation(BC, X_train_scaled, y_train)

