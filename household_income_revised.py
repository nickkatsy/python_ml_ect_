import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('C:/ml/python/data/income.csv',delimiter=',')


df.info()


df.columns = df.columns.str.replace('.','_')


df.isna().sum()

df.nunique()

df['income'] = [0 if X == '<=50K' else 1 for X in df['income']]


print(df.dtypes)

df1 = df.copy()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for i in df1:
    df1[i] = le.fit_transform(df1[i])



import seaborn as sns
import matplotlib.pyplot as plt


plt.figure(figsize=(10,6))
sns.heatmap(df1.corr(),annot=True)



def subplots(df1):
    plt_,axs = plt.subplots(2,2,figsize=(15, 8),gridspec_kw={'hspace':0.5})

    sns.barplot(x='education',y='income',data=df1, ax=axs[0,0])
    axs[0,0].set_title('Education vs. Income (0: <= 50K, 1: > 50K)')
    axs[0,0].set_xlabel('Education Level')
    axs[0,0].set_ylabel('Income')

    sns.kdeplot(x='age', y='hours_per_week', hue='education',data=df1,fill=True,ax=axs[0,1])
    axs[0,1].set_title('Age vs. Hours per Week by Education')
    axs[0,1].set_xlabel('Age')
    axs[0,1].set_ylabel('Hours per Week')

    sns.countplot(x='marital_status',hue='income',data=df1,ax=axs[1,0])
    axs[1,0].set_title('Marital Status (0: <= 50K, 1: > 50K)')
    axs[1,0].set_xlabel('Marital Status')
    axs[1,0].set_ylabel('Count')

    sns.boxplot(x='income',y='age',data=df1,ax=axs[1,1])
    axs[1,1].set_title('Age vs. Income (0: <= 50K, 1: > 50K)')
    axs[1,1].set_xlabel('Income')
    axs[1,1].set_ylabel('Age')

    plt.show()

subplots(df1)





X = df.drop('income',axis=1)
y = df['income']



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()


from sklearn.compose import make_column_transformer

ct = make_column_transformer(
    (ohe,X.select_dtypes(include='object').columns),
    (sc,X.select_dtypes(include=['int64','float64']).columns),remainder='passthrough')


ct.fit_transform(X)



from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
rfc = RandomForestClassifier()
gbc = GradientBoostingClassifier()

from sklearn.tree import DecisionTreeClassifier
trees = DecisionTreeClassifier()


from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()


from sklearn.pipeline import make_pipeline

lr_pipe = make_pipeline(ct,lr).fit(X_train,y_train)
lr_pred = lr_pipe.predict(X_test)
lr_pred_prob = lr_pipe.predict_proba(X_test)[::,1]

rfc_pipe = make_pipeline(ct,rfc).fit(X_train,y_train)
rfc_pred = rfc_pipe.predict(X_test)
rfc_pred_prob = rfc_pipe.predict_proba(X_test)[::,1]

gbc_pipe = make_pipeline(ct,gbc).fit(X_train,y_train)
gbc_pred = gbc_pipe.predict(X_test)
gbc_pred_prob = gbc_pipe.predict_proba(X_test)[::,1]


nb_pipe = make_pipeline(ct,lda).fit(X_train,y_train)
nb_pred = nb_pipe.predict(X_test)
nb_pred_prob = nb_pipe.predict_proba(X_test)[::,1]


lda_pipe = make_pipeline(ct,lda).fit(X_train,y_train)
lda_pred = lda_pipe.predict(X_test)
lda_pred_prob = lda_pipe.predict_proba(X_test)[::,1]


tree_pipe = make_pipeline(ct,trees).fit(X_train,y_train)
tree_pred = tree_pipe.predict(X_test)
tree_pred_prob = tree_pipe.predict_proba(X_test)[::,1]


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=10)
knn_pipe = make_pipeline(ct,knn).fit(X_train,y_train)
knn_pred = knn_pipe.predict(X_test)
knn_pred_prob = knn_pipe.predict_proba(X_test)[::,1]



from sklearn.metrics import roc_auc_score,accuracy_score,roc_curve

def evaluate_model(y_test,y_pred,y_pred_prob,model_name):
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred_prob)
    print(f'{model_name} --Accuracy Score-- {acc*100:.2f}%, --ROC-- {roc*100:.2f}%')



evaluate_model(y_test, lr_pred, lr_pred_prob, 'Logistic Regression')
evaluate_model(y_test, rfc_pred, rfc_pred_prob, 'Random Forest')
evaluate_model(y_test, gbc_pred, gbc_pred_prob, 'Gradient Boost')
evaluate_model(y_test, nb_pred, nb_pred_prob, 'Naive Bayes')
evaluate_model(y_test, lda_pred, lda_pred_prob, 'LDA')
evaluate_model(y_test, knn_pred, knn_pred_prob, 'K-NearestNeighbors')
evaluate_model(y_test, tree_pred, tree_pred_prob, 'Decision Trees')

def ROC(y_test,y_pred_prob,model_name):
    fpr,tpr, _ = roc_curve(y_test,y_pred_prob)
    plt.plot(fpr,tpr,label=model_name)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    


ROC(y_test, lr_pred_prob,'Logistic Regression')
ROC(y_test,gbc_pred_prob,'Gradient Boosting')
ROC(y_test, rfc_pred_prob, 'Random Forest')
ROC(y_test,lda_pred_prob,'LDA')
ROC(y_test, nb_pred_prob, 'Naive Bayes')
ROC(y_test, knn_pred_prob, 'K-Nearest Neighbors')
ROC(y_test,tree_pred_prob,'Decision Trees')
plt.legend()
plt.show()

# GridSearch on all parameters
# WARNING: This will take a long time
from sklearn.model_selection import RandomizedSearchCV

import numpy as np


lr_random_param_grid = {
    'logisticregression__C': np.logspace(-4,4,9),
    'logisticregression__penalty': ['l1','l2']
}


lr_random_search = RandomizedSearchCV(
    lr_pipe,lr_random_param_grid,n_iter=10,scoring='roc_auc',cv=5,random_state=42
).fit(X_train, y_train)

# Best Logistic Regression model
best_lr_estimator = lr_random_search.best_estimator_
lr_random_pred = best_lr_estimator.predict(X_test)
lr_random_pred_prob = best_lr_estimator.predict_proba(X_test)[::,1]
print(f'Logistic Regression - Best Parameters: {lr_random_search.best_params_}')
print(f'Logistic Regression - Best ROC-AUC Score: {lr_random_search.best_score_ * 100:.2f}%')


# Random forest Random Grid Search
rfc_random_param_grid = {
    'randomforestclassifier__n_estimators': [50,100,200],
    'randomforestclassifier__max_depth': [None,10,20],
    'randomforestclassifier__min_samples_split': [2,5,10],
    'randomforestclassifier__min_samples_leaf': [1,2,4]
}

rfc_random_search = RandomizedSearchCV(
    rfc_pipe, rfc_random_param_grid,n_iter=10,scoring='roc_auc',cv=5,random_state=42
).fit(X_train,y_train)


best_rfc_estimator = rfc_random_search.best_estimator_
rfc_random_pred = best_rfc_estimator.predict(X_test)
rfc_random_pred_prob = best_rfc_estimator.predict_proba(X_test)[::,1]
print(f'Random Forest - Best Parameters: {rfc_random_search.best_params_}')
print(f'Random Forest - Best ROC-AUC Score: {rfc_random_search.best_score_ * 100:.2f}%')


#Gradientboost gridsearch

gbc_random_param_grid = {
    'gradientboostingclassifier__n_estimators': [50,100,200],
    'gradientboostingclassifier__learning_rate': [0.01,0.1,0.2],
    'gradientboostingclassifier__max_depth': [3,4,5],
    'gradientboostingclassifier__min_samples_split': [2,5,10],
}

gbc_random_search = RandomizedSearchCV(
    gbc_pipe,gbc_random_param_grid,n_iter=10,scoring='roc_auc',cv=5,random_state=42
).fit(X_train,y_train)


# Best Gradient Boosting model using random gridsearch
best_gbc_estimator = gbc_random_search.best_estimator_
gbc_random_pred = best_gbc_estimator.predict(X_test)
gbc_random_pred_prob = best_gbc_estimator.predict_proba(X_test)[::,1]
print(f'Gradient Boosting - Best Parameters: {gbc_random_search.best_params_}')
print(f'Gradient Boosting - Best ROC-AUC Score: {gbc_random_search.best_score_ * 100:.2f}%')


# KNN Random Grid Search

knn_param_grid = {
    'kneighborsclassifier__n_neighbors': [3,5,7,10],
    'kneighborsclassifier__weights': ['uniform','distance'],
    'kneighborsclassifier__p': [1,2]
}

knn_grid_search = RandomizedSearchCV(knn_pipe,knn_param_grid,scoring='roc_auc',cv=5).fit(X_train, y_train)


best_knn_estimator = knn_grid_search.best_estimator_
knn_grid_pred = best_knn_estimator.predict(X_test)
knn_grid_pred_prob = best_knn_estimator.predict_proba(X_test)[::,1]
print(f'K-Nearest Neighbors - Best Parameters: {knn_grid_search.best_params_}')
print(f'K-Nearest Neighbors - Best ROC-AUC Score: {knn_grid_search.best_score_ * 100:.2f}%')

#Trees

dt_param_dist = {
    'decisiontreeclassifier__criterion': ['gini','entropy'],
    'decisiontreeclassifier__max_depth': [None,10,20,30],
    'decisiontreeclassifier__min_samples_split': [2,5,10],
    'decisiontreeclassifier__min_samples_leaf': [1,2,4]
}

dt_rand_search = RandomizedSearchCV(
    tree_pipe,dt_param_dist,n_iter=10,scoring='roc_auc',cv=5,random_state=42
).fit(X_train,y_train)


best_dt_estimator = dt_rand_search.best_estimator_
dt_rand_pred = best_dt_estimator.predict(X_test)
dt_rand_pred_prob = best_dt_estimator.predict_proba(X_test)[::,1]
print(f'Decision Trees - Best Parameters: {dt_rand_search.best_params_}')
print(f'Decision Trees - Best ROC-AUC Score: {dt_rand_search.best_score_ * 100:.2f}%')

#LDA

lda_param_dist = {
    'lineardiscriminantanalysis__solver': ['svd','lsqr','eigen'],
    'lineardiscriminantanalysis__shrinkage': [None,'auto']
}

lda_rand_search = RandomizedSearchCV(
    lda_pipe,lda_param_dist,n_iter=10,scoring='roc_auc',cv=5,random_state=42
)
lda_rand_search.fit(X_train, y_train)

best_lda_estimator = lda_rand_search.best_estimator_
lda_rand_pred = best_lda_estimator.predict(X_test)
lda_rand_pred_prob = best_lda_estimator.predict_proba(X_test)[::,1]
print(f'Linear Discriminant Analysis - Best Parameters: {lda_rand_search.best_params_}')
print(f'Linear Discriminant Analysis - Best ROC-AUC Score: {lda_rand_search.best_score_ * 100:.2f}%')


# This took very,very long.
# ROC Curves for Hyper-Parameters
ROC(y_test,lr_random_pred_prob, 'Logistic Regression Random CV')
ROC(y_test,rfc_random_pred_prob,'Random Forest Random Grid Search CV')
ROC(y_test,gbc_random_pred_prob,'Gradient Boost Random Grid Search CV')
ROC(y_test,dt_rand_pred_prob,'Decision Tree Random Grid Search CV')
ROC(y_test,lda_rand_pred_prob,'LDA Random Grid Search CV')
plt.legend()
plt.show()
