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

def basic_subplots(dataframe):
  plt____,axs = plt.subplots(2,2,figsize=(12,6))
  sns.scatterplot(df,x='income',y='house_price',ax=axs[0,0])
  sns.kdeplot(df,x='in_cali',ax=axs[0,1])
  sns.violinplot(df['income'],ax=axs[1,0])
  sns.distplot(df['earthquake'],ax=axs[1,1])
  plt.show()


basic_subplots(df)

# Selection

X = df.drop('in_cali',axis=1)
y = df[['in_cali']]


# scaling the features

from sklearn.preprocessing import StandardScaler


sc = StandardScaler()
X_scaler = sc.fit_transform(X)
y_scaler = sc.fit_transform(y)

# Creating Classifier
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression().fit(X_scaler,y_scaler)

clf_pred = clf.predict(X_scaler)

clf_pred_prob = clf.predict_proba(X_scaler)[::,1]

# 5-Fold CV
from sklearn.model_selection import cross_val_score

cv = cross_val_score(clf,X_scaler,y_scaler,cv=5,scoring='roc_auc').mean()
print(cv*100)

