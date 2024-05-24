import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('C:/ML/python/data/animequotes.csv',delimiter=',')


nltk.download('punkt')


def text_tokenizer(text):
    return nltk.word_tokenize(text)


tfid = TfidfVectorizer(tokenizer=text_tokenizer)



df['Quote'] = df['Quote'].str.lower()
df['Quote'] = df['Quote'].str.replace("W","")
df['Character'] = df['Character'].str.replace("W","")
df['Anime'] = df['Anime'].str.replace("W","")


X = df['Quote']
y = df['Character']


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)

X_train = tfid.fit_transform(X_train)
X_test = tfid.transform(X_test)

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier,BaggingClassifier


from sklearn.svm import SVC

model = {
    "GBC":GradientBoostingClassifier(),
    "BC":BaggingClassifier(),
    "LR":LogisticRegression(),
    "svc":SVC(probability=True)
    }




from sklearn.metrics import accuracy_score,classification_report

for name,model in model.items():
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    clf_rpt = classification_report(y_test, y_pred)
    print(f'{model} --ACC-- {acc*100:.2f}%; --Classifcation report-- {clf_rpt}')




#This shit is hard and pretty worthless
