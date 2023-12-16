import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('https://raw.githubusercontent.com/nickkas/python_ml_ect_/master/heart.csv')

df.info()

df.isna().sum()

print(df.shape)

df.dtypes

df['heart_attack'] = df['target']

df = df.drop('target',axis=1)



import seaborn as sns
import matplotlib.pyplot as plt


sns.heatmap(df.corr(), annot=True)


def subplots(df):
    plt_,ax1 = plt.subplots(3,3,figsize=(12,5))
    sns.histplot(df[['age']],ax=ax1[0,0])
    sns.violinplot(df[['cp']],ax=ax1[0,1])  
    sns.histplot(df[['trestbps']],ax=ax1[0,2])
    sns.histplot(df[['fbs']],ax=ax1[1,0])
    sns.histplot(df[['sex']],ax=ax1[1,1])
    sns.histplot(df['exang'],ax=ax1[1,2])
    sns.histplot(df[['heart_attack']],ax=ax1[2,0])
    sns.histplot(df[['chol']],ax=ax1[2,1])
    sns.kdeplot(df[['oldpeak']],ax=ax1[2,2])
    plt.show()



subplots(df)


import statsmodels.api as sm

model_sex = sm.GLM(exog=sm.add_constant(df[['sex']]),endog=df[['heart_attack']]).fit()
print(model_sex.summary())


model_age = sm.OLS(exog=sm.add_constant(df['age']),endog=df[['heart_attack']]).fit()
print(model_age.summary())

model_chol = sm.OLS(exog=sm.add_constant(df[['chol']]),endog=df[['heart_attack']]).fit()
print(model_chol.summary())

#for full model
y_endog = df['heart_attack']
X_exog = sm.add_constant(df.drop('heart_attack', axis=1))

full_model = sm.GLM(y_endog,X_exog,family=sm.families.Binomial()).fit()
print(full_model.summary())


coefficients = pd.DataFrame({'Feature': X_exog.columns,'Coefficient': full_model.params.values})

# Display coefficients
print(coefficients)



#Feature Selection

X = df.drop('heart_attack',axis=1)
y = df[['heart_attack']]

y.value_counts(normalize=True)





from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier



clf = LogisticRegression().fit(X_train_scaled,y_train)
clf_pred = clf.predict(X_test_scaled)
clf_pred_prob = clf.predict_proba(X_test_scaled)[::,1]

rf = RandomForestClassifier().fit(X_train_scaled,y_train)
rf_pred = rf.predict(X_test_scaled)
rf_pred_prob = rf.predict_proba(X_test_scaled)[::,1]


knn = KNeighborsClassifier(n_neighbors=7).fit(X_train_scaled,y_train)
knn_pred = knn.predict(X_test_scaled)
knn_pred_prob = knn.predict_proba(X_test_scaled)[::,1]



nb = GaussianNB().fit(X_train_scaled,y_train)
nb_pred = nb.predict(X_test_scaled)
nb_pred_prob = nb.predict_proba(X_test_scaled)[::,1]


GBC = GradientBoostingClassifier().fit(X_train_scaled,y_train)
GBC_pred = GBC.predict(X_test_scaled)
GBC_pred_prob = GBC.predict_proba(X_test_scaled)[::,1]



from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score

def evaluate_model(model_name,y_true,y_pred,y_pred_prob):
    acc = accuracy_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_pred_prob)
    print(f'{model_name} - Accuracy: {acc * 100:.2f}%, ROC-AUC: {roc * 100:.2f}%')

evaluate_model('Logistic Regression', y_test,clf_pred,clf_pred_prob)
evaluate_model('Random Forest', y_test,rf_pred,rf_pred_prob)
evaluate_model('Naive Bayes', y_test,nb_pred,nb_pred_prob)
evaluate_model('Gradient Boosting',y_test,GBC_pred,GBC_pred_prob)
evaluate_model('KNN', y_test,knn_pred, knn_pred_prob)



def roc_curve_plot(y_true, y_pred_prob, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    plt.plot(fpr,tpr,label=model_name)
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

roc_curve_plot(y_test,clf_pred_prob,'Logistic Regression')
roc_curve_plot(y_test,rf_pred_prob,'Random Forest')
roc_curve_plot(y_test,knn_pred_prob,'KNN')
roc_curve_plot(y_test,nb_pred_prob,'Naive Bayes')
roc_curve_plot(y_test,GBC_pred_prob,'Gradient Boosting')
plt.legend()
plt.show()
