import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')



app = pd.read_csv('https://raw.githubusercontent.com/LeeMorinUCF/QMB6358F23/main/final_exam_2021/applications.csv')
dem = pd.read_csv('https://raw.githubusercontent.com/LeeMorinUCF/QMB6358F23/main/final_exam_2021/demographic.csv')
credit = pd.read_csv('https://raw.githubusercontent.com/LeeMorinUCF/QMB6358F23/main/final_exam_2021/credit_bureau.csv')



for i in [app,dem,credit]:
    print(i.nunique())
    print(i.info())
    print(i.isna().sum())
    print(i.corr())



app['homeownership'] = pd.get_dummies(app.homeownership,prefix='homeownership').iloc[::,1]


for i in [credit,app]:
    print(i.nunique())
    print(i.corr())
    
    


df = pd.merge(credit,app)



df.isna().sum()

import seaborn as sns


def subplots(df):
    plt_,axs = plt.subplots(2,2,figsize=(10,8))
    sns.scatterplot(df,x='credit_limit',y='purchases',ax=axs[0,0])
    sns.scatterplot(df,x='app_id',y='purchases',ax=axs[0,1])
    sns.kdeplot(df,x='homeownership',y='past_def',ax=axs[1,0])
    sns.kdeplot(df[['credit_limit']],ax=axs[1,1])
    plt.show()

subplots(df)



X = df.drop('homeownership',axis=1)
y = df[['homeownership']]

y.value_counts(normalize=True)


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(X_train)

X_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression().fit(X_scaled,y_train)

clf_pred = clf.predict(X_test_scaled)

clf_pred_prob = clf.predict_proba(X_test_scaled)[::,1]


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier().fit(X_scaled,y_train)
rfc_pred = rfc.predict(X_test_scaled)
rfc_pred_prob = rfc.predict_proba(X_test_scaled)[::,1]


from sklearn.naive_bayes import GaussianNB

nb = GaussianNB().fit(X_scaled,y_train)
nb_pred = nb.predict(X_test_scaled)
nb_pred_prob = nb.predict_proba(X_test_scaled)[::,1]

from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier().fit(X_scaled,y_train)
gbc_pred = gbc.predict(X_test_scaled)
gbc_pred_prob = gbc.predict_proba(X_test_scaled)[::,1]




from sklearn.metrics import roc_auc_score,accuracy_score

clf_acc = accuracy_score(y_test, clf_pred)
print(f'the accuracy of the logistic regression model: {clf_acc}')

roc_clf = roc_auc_score(y_test, clf_pred_prob)
print(f'the roc_auc score for the logistic regression model: {roc_clf}')


# metrics for random forest

acc_rfc = accuracy_score(y_test, rfc_pred)
print(f'the accuracy using random forrest classification is: {acc_rfc}')

roc_rfc = roc_auc_score(y_test, rfc_pred_prob)
print(f'the roc auc score using random forest: {roc_rfc}')


# metrics for naive_bayes

acc_bayes = accuracy_score(y_test, nb_pred)
print(f'the accuracy using naive bayes: {acc_bayes}')

roc_bayes = roc_auc_score(y_test, nb_pred_prob)
print(f'the roc score using naive bayes is: {roc_bayes}')


# metrics from gradient boosting

acc_grad = accuracy_score(y_test, gbc_pred)
print(f'the accuracy using gradientboosting: {acc_grad}')


roc_grad = roc_auc_score(y_test, gbc_pred_prob)
print(f'the roc score using gradient boosing is: {roc_grad}')





df_merged = pd.merge(df,dem)

df_merged.info()

df_merged.corr()


df_merged = df_merged.drop(['ssn','zip_code'],axis=1)

df_merged.isna().sum()

def merged_date(df_merged):
    plt_2,axs2 = plt.subplots(2,2,figsize=(10,6))
    sns.kdeplot(df_merged,x='fico',ax=axs2[0,0])
    sns.scatterplot(df_merged,x='app_id',y='income',ax=axs2[0,1])
    sns.barplot(df_merged,x='num_late',y='num_bankruptcy',ax=axs2[1,0])
    sns.barplot(df_merged,x='num_late',y='homeownership',ax=axs2[1,1])
    plt.show()

merged_date(df_merged)



df_merged['utility'] = df_merged['credit_limit'] * (df_merged['income']/df_merged['purchases'])


print(df_merged['utility'].describe())

df_merged.columns

def utility_merged_subplots(df_merged):
    plt_,axs3 = plt.subplots(2,2,figsize=(10,6))
    sns.barplot(df_merged,x='num_late',y='utility',ax=axs3[0,0])
    sns.scatterplot(df_merged,x='avg_income',y='utility',ax=axs3[0,1])
    sns.scatterplot(df_merged,x='credit_limit',y='utility',ax=axs3[1,0])
    sns.barplot(df_merged,x='num_late',y='past_def',ax=axs3[1,1])
    plt.show()
    

utility_merged_subplots(df_merged)




X_u = df_merged.drop(['utility'], axis=1)
y_u = df_merged['utility']

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
Xu_poly = poly.fit_transform(X_u)


scaler_poly = StandardScaler()
Xu_poly_scaled = scaler_poly.fit_transform(Xu_poly)


from sklearn.linear_model import LinearRegression


lr_utility = LinearRegression().fit(Xu_poly_scaled,y_u)


from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(lr_utility,Xu_poly_scaled,y_u,cv=10,scoring='r2')

print(f'Max R-squared (cross-validated): {cv_scores.max()}')
