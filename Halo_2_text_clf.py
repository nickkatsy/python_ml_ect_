import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('https://raw.githubusercontent.com/nickkatsy/python_ml_ect_/master/arbiter.csv')

nltk.download('punkt')


df['line'] = df['line'].str.replace('W',"")
df['name'] = df['name'].str.replace("W","")


X = df['line']
y = df['name']



def tokenizer_text(text):
    return nltk.word_tokenize(text)


tfid = TfidfVectorizer(tokenizer=tokenizer_text)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)


X_train = tfid.fit_transform(X_train)
X_test = tfid.transform(X_test)

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import BaggingClassifier,GradientBoostingClassifier

from sklearn.svm import SVC

models = {
    "GBC":GradientBoostingClassifier(),
    "LR":LogisticRegression(),
    "BC":BaggingClassifier(),
    "svc":SVC(probability=True)
    }


from sklearn.metrics import accuracy_score,classification_report

for name,models in models.items():
    models.fit(X_train,y_train)
    y_pred = models.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    clf_rpt = classification_report(y_test, y_pred)
    print(f'{models} -- ACC -- {acc*100:.2f}%; --Classification Report-- {clf_rpt}')
