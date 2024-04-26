import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('https://raw.githubusercontent.com/nickkatsy/python_ml_ect_/master/Spam%20Email%20raw%20text%20for%20NLP.csv')

df.info()

df['MESSAGE'] = df['MESSAGE'].str.lower()
df['MESSAGE'] = df['MESSAGE'].str.replace('[^\w\s]','')


X = df['MESSAGE']
y = df['CATEGORY']


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)

tfid = TfidfVectorizer()

X_tfid_train = tfid.fit_transform(X_train)
X_tfid_test = tfid.transform(X_test)


from sklearn.naive_bayes import MultinomialNB

NB = MultinomialNB().fit(X_tfid_train,y_train)
NB_pred = NB.predict(X_tfid_test)


from sklearn.metrics import classification_report,accuracy_score

clr_rpt = classification_report(y_test,NB_pred)
print('clr_rpt',clr_rpt*100)
acc = accuracy_score(y_test,NB_pred)
print('accuracy: ',acc*100)



from sklearn.ensemble import GradientBoostingClassifier


GBC = GradientBoostingClassifier().fit(X_tfid_train,y_train)
GBC_pred = GBC.predict(X_tfid_test)

acc_gbc = accuracy_score(y_test,GBC_pred)
print(f'accuracy using Gradient Boosting Classifier: {acc_gbc*100:.2f}%')
clfrpt_gbc = classification_report(y_test, GBC_pred)
print('clf report using gradient boosting clf',clfrpt_gbc)







