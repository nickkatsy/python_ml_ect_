from nltk.corpus import stopwords
import pandas as pd
import spacy
import re
import string
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder




df = pd.read_csv('C:/ML/python/data/arbiter.csv',delimiter=',')


df['line'] = df['line'].str.lower()
df['name'] = df['name'].str.lower()




def remove_html_tags(text):
    pattern = r'[^a-zA-Z0-9\s]'
    text = re.sub(pattern,'',text)
    return text

df['line'] = df['line'].apply(remove_html_tags)
print(df['line'])
print(df['name'])


#removing punctions using strings


def remove_punctuation(text):
    return text.translate(str.maketrans('','',string.punctuation))

df['line'] = df['line'].apply(remove_punctuation)




#removing stopwords

stop_words = set(stopwords.words('english'))
print(stop_words)
def remove_stopwords(text):
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]

    text = ' '.join(tokens)
    return text


df['line'] = df['line'].apply(remove_stopwords)

nlp = spacy.load("en_core_web_sm")



#I chose stemming over lemmentization


def lemmatize_text(text):
    doc = nlp(text)
    lemmatized_text = ' '.join([token.lemma_ for token in doc])
    return lemmatized_text

df['line'] = df['line'].apply(lemmatize_text)



from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

X = df['line']
y = df['name']

le = LabelEncoder()

X = cv.fit_transform(X)
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)





from sklearn.naive_bayes import MultinomialNB
MNB = MultinomialNB()

from sklearn.svm import SVC
svc = SVC(probability=True)

from sklearn.ensemble import GradientBoostingClassifier
GBC = GradientBoostingClassifier()
from sklearn.metrics import accuracy_score,classification_report

def evaluate_models(X_train,X_test,y_train,y_test,model):
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test,pred)
    clf_rpt = classification_report(y_test,pred)
    print(f'{model.__class__.__name__}, --ACC-- {acc*100:.2f}%; --Classification Report-- {clf_rpt}%')
    return pred



MNB_pred = evaluate_models(X_train, X_test, y_train, y_test, MNB)
svc_pred = evaluate_models(X_train, X_test, y_train, y_test, svc)
GBC_pred = evaluate_models(X_train, X_test, y_train, y_test, GBC)







