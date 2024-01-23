import pandas as pd
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('C:/ML/python/data/labor_supply.csv',delimiter=',',nrows=10000)


df.isna().sum()
print(df.dtypes)
df.nunique()
df['SEX'].value_counts()
df['SEX'] = pd.get_dummies(df.SEX,prefix='SEX').iloc[:,0:1]
woman_with_young_children = df[(df['SEX'] == 0) & (df['NCHLT5'] == 1)]
woman_without_young_children = df[(df['SEX'] == 0) & (df['NCHLT5'] != 1)]


#laborforce variable

df['LABFORCE'] = [1 if X == 2 else 0 for X in df['LABFORCE']]


# Men with and without young children
men_with_young_children = df[(df['SEX'] == 1) & (df['NCHLT5'] == 1)]
men_without_young_children = df[(df['SEX'] == 1) & (df['NCHLT5'] != 1)]

import seaborn as sns
import matplotlib.pyplot as plt


sns.heatmap(df.corr(),annot=True)


def subplots(df):
    _,axs = plt.subplots(2,2,figsize=(10,6))
    sns.kdeplot(x='YEAR',ax=axs[0,0],data=df)
    sns.boxplot(x='LABFORCE',y='YEAR',ax=axs[0,1],data=df)
    sns.boxplot(x='YEAR',y='AGE',ax=axs[1,0],data=df)
    sns.lineplot(x='YEAR',y='LABFORCE',ax=axs[1,1],data=df)
    plt.show()

subplots(df)






def Woman_with_children(df):
    sns.kdeplot(data=woman_with_young_children,x='AGE',y='YEAR',fill=True,cmap='Blues',cbar=True)
    plt.title('Age vs Women with Children under the Age of 5')
    plt.xlabel('Age')
    plt.ylabel('Year')
    plt.show()

Woman_with_children(df)


def women_with_no_children(df):
    sns.scatterplot(x='AGE',y='YEAR',data=woman_without_young_children,marker='o',color='blue',alpha=0.5)
    sns.regplot(x='AGE',y='YEAR',data=woman_without_young_children,scatter=False,color='red')
    plt.xlabel('AGE')
    plt.ylabel('YEAR')
    plt.title('Women Without Young children by year')
    plt.show()


women_with_no_children(df)


def men_with_children(df):
    sns.kdeplot(data=men_with_young_children, x='AGE',y='YEAR',fill=True,cmap='Reds',cbar=True)
    plt.title('Age vs Men with Children under the Age of 5')
    plt.xlabel('Age of Dude with Child')
    plt.ylabel('Year')
    plt.show()

men_with_children(df)

def men_with_no_children(df):
    sns.scatterplot(x='YEAR',y='AGE',data=men_without_young_children,alpha=0.5)
    sns.regplot(x='YEAR',y='AGE', data=men_without_young_children,scatter=False,color='red')
    plt.title('Men Without Young Children by Year')
    plt.xlabel('Age of Male With no Child')
    plt.ylabel('Year')
    plt.show()

men_with_no_children(df)



import statsmodels.api as sm

X_sex = sm.add_constant(df['SEX'])
y = df['LABFORCE']

model = sm.OLS(exog=X_sex,endog=y).fit()
print(model.summary())

X_year = sm.add_constant(df['YEAR'])
model_year = sm.OLS(exog=X_year,endog=y).fit()
print(model_year.summary())


#all together

X = df.drop(['LABFORCE','ASECWT'],axis=1)
y = df['LABFORCE']

model_full = sm.OLS(exog=sm.add_constant(X),endog=y).fit()
print(model_full.summary())


df['LABFORCE'].nunique()



from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)


from sklearn.preprocessing import MinMaxScaler

ms = MinMaxScaler()

X_train_scaled = ms.fit_transform(X_train)
X_test_scaled = ms.transform(X_test)


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,BaggingClassifier
rfc = RandomForestClassifier()
GBC = GradientBoostingClassifier()
BC = BaggingClassifier()

from sklearn.svm import SVC
svc = SVC(probability=True)


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()


from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score

def evaluate_model(model,X_train_scaled,X_test_scaled,y_train,y_test):
    model = model.fit(X_train_scaled,y_train)
    pred = model.predict(X_test)
    pred_prob = model.predict_proba(X_test_scaled)[:,1]
    acc = accuracy_score(y_test, pred)
    roc = roc_auc_score(y_test, pred_prob)
    print(f'{model.__class__.__name__}, --ACC-- {acc*100:.2f}%; --ROC-- {roc*100:.2f}%')
    return pred,pred_prob



lr_pred,lr_pred_prob = evaluate_model(lr, X_train_scaled, X_test_scaled, y_train, y_test)
GBC_pred,GBC_pred_prob = evaluate_model(GBC, X_train_scaled, X_test_scaled, y_train, y_test)
BC_pred_prob = evaluate_model(GBC, X_train_scaled, X_test_scaled, y_train, y_test)
svc_pred_prob = evaluate_model(GBC, X_train_scaled, X_test_scaled, y_train, y_test)
lda_pred,lda_pred_prob = evaluate_model(lda, X_train_scaled, X_test_scaled, y_train, y_test)




