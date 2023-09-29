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


plt____,axs = plt.subplots(2,2,figsize=(12,6))
plt_1 = sns.scatterplot(df,x='income',y='house_price',ax=axs[0,0])
plt_2 = sns.barplot(df,x='in_cali',ax=axs[0,1])
plt_3 = sns.violinplot(df['income'],ax=axs[1,0])
plt_4 = sns.distplot(df[['earthquake']],ax=axs[1,1])




# Just a simple model for stuuuuuuuuf, in_cali for now I do not know I do no know atm

X = df.drop('in_cali',axis=1)
y = df[['in_cali']]



from sklearn.preprocessing import StandardScaler


sc = StandardScaler()
X_scaler = sc.fit_transform(X)
y_scaler = sc.fit_transform(y)


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression().fit(X_scaler,y_scaler)

clf_pred = clf.predict(X_scaler)

clf_pred_prob = clf.predict_proba(X_scaler)[::,1]


from sklearn.model_selection import cross_val_score

cv = cross_val_score(clf,X_scaler,y_scaler,cv=5,scoring='roc_auc').mean()
print(cv*100)

