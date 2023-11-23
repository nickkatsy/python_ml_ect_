import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('https://raw.githubusercontent.com/nickkas/python_ml_ect_/master/diabetes.csv')


df.info()
df.isna().sum()
df.nunique()
df.dtypes





def desc_plots(df):
    plt_,axs = plt.subplots(2,4,figsize=(10,6))
    sns.boxplot(df[['Pregnancies']],ax=axs[0,0])
    sns.boxplot(df[['Glucose']],ax=axs[0,1])
    sns.boxplot(df[['BloodPressure']],ax=axs[0,2])
    sns.boxplot(df[['SkinThickness']],ax=axs[0,3])
    sns.boxplot(df[['Insulin']],ax=axs[1,0])
    sns.boxplot(df[['BMI']],ax=axs[1,1])
    sns.boxplot(df[['DiabetesPedigreeFunction']],ax=axs[1,2])
    sns.boxplot(df[['Age']],ax=axs[1,3])
    plt.show()



desc_plots(df)


plt.figure(figsize=(12,4))
sns.heatmap(df.corr(),annot=True)


# descriptive statisics

import statsmodels.api as sm

y_endog = df['Outcome']
X_exog = sm.add_constant(df.drop('Outcome', axis=1))

full_model = sm.GLM(y_endog,X_exog,family=sm.families.Binomial()).fit()
print(full_model.summary())


coefficients = pd.DataFrame({'Feature': X_exog.columns,'Coefficient': full_model.params.values})
print(coefficients)

X = df.drop('Outcome',axis=1)
y = df[['Outcome']]

y.value_counts(normalize=True)



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit_transform(X)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB


clf = LogisticRegression().fit(X,y)
clf_pred = clf.predict(X)
clf_pred_prob = clf.predict_proba(X)[::,1]

rfc = RandomForestClassifier().fit(X,y)
rfc_pred = rfc.predict(X)
rfc_pred_prob = rfc.predict_proba(X)[::,1]

Knn = KNeighborsClassifier(n_neighbors=6).fit(X,y)
Knn_pred = Knn.predict(X)
Knn_pred_prob = Knn.predict_proba(X)[::,1]

nb = GaussianNB().fit(X,y)
nb_pred = nb.predict(X)
nb_pred_prob = nb.predict_proba(X)[::,1]

gb = GradientBoostingClassifier().fit(X,y)
gb_pred = gb.predict(X)
gb_pred_prob = gb.predict_proba(X)[::,1]


from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score
from sklearn.model_selection import cross_val_score

# Logistic Regression Results
clf_acc = accuracy_score(y, clf_pred)
print('Accuracy of Logistic Regression Model',clf_acc*100)

clf_roc = roc_auc_score(y, clf_pred_prob)
print('logistic Regression ROC: ',clf_roc*100)

#Logistic Regression Model produces a higher roc with 10 fold cross-validation
cv_clf = cross_val_score(clf, X,y,cv=10,scoring='roc_auc').max()
print('Logistic Regression ROC with Cross-Validation',cv_clf*100)


#Random Forest Classification Results

acc_rfc = accuracy_score(y,rfc_pred)
print('accuracy score Random Forest Classification',acc_rfc*100)

roc_rfc = roc_auc_score(y,rfc_pred_prob)
print('roc using Random Forest Classification= ',roc_rfc)

# KNN Classification Results

acc_knn = accuracy_score(y, Knn_pred)
print('accuarcy score using knn= ',acc_knn*100)

roc_knn = roc_auc_score(y, Knn_pred_prob)
print('roc using knn',roc_auc_score(y, Knn_pred_prob)*100)

# Results from Naive Bayes Classification

acc_nb = accuracy_score(y, nb_pred)
print(f'the Accuracy of the Naive Bayes Classifier: {acc_nb}')

roc_nb = roc_auc_score(y,nb_pred_prob)
print(f'the roc/auc score from the Naive Bayes Classifier: {roc_nb}')

acc_gb = accuracy_score(y, gb_pred)
print(f'the accuracy using Gradient boost classifier: {acc_gb}')

roc_gb = roc_auc_score(y,gb_pred_prob)
print(f'the roc_auc score using Gradient Boost classifier: {roc_gb}')

# ROC Curves plotted
def roc_curve_plot(y, y_pred_prob,model_name):
    fpr, tpr, _ = roc_curve(y, y_pred_prob)
    plt.plot(fpr,tpr,label=model_name)
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    
    
roc_curve_plot(y,clf_pred_prob,'Logistic Regression')
roc_curve_plot(y,rfc_pred_prob,'Random Forest')
roc_curve_plot(y,nb_pred_prob,'Naive Bayes')
roc_curve_plot(y,gb_pred_prob,'Gradient Boosting')
plt.legend()
plt.show()
