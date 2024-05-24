import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import nltk
nltk.download('stopwords')


df = pd.read_csv('C:/ML/python/data/Capsule Hotel Risk Classification Dataset (English Translated) - Sheet1.csv',delimiter=',')

nltk.download('punkt')
df.info()
df = df.drop('Unnamed: 0',axis=1)

df['content'] = df['content'].apply(lambda x: x.lower())
df['translated_content'] = df['translated_content'].apply(lambda x: x.lower())


import re


def remove_html_tags(text):
    pattern = r'[^a-zA-z0-1\s]'
    text = re.sub(pattern,'',text)
    return text

df['content'] = df['content'].apply(remove_html_tags)
df['translated_content'] = df['translated_content'].apply(remove_html_tags)


import string

punc = string.punctuation


def remove_punc(text):
    return text.translate(str.maketrans('', '', punc))


df['content'] = df['content'].apply(remove_punc)
df['translated_content'] = df['translated_content'].apply(remove_punc)

from nltk.tokenize import word_tokenize

df['content'] = df['content'].apply(lambda x: word_tokenize(x))
df['translated_content'] = df['translated_content'].apply(lambda x: word_tokenize(x))










w = nltk.WordNetLemmatizer()
def lemmatization_join(tokenized_text):
    text_lemma = [w.lemmatize(word) for word in tokenized_text]
    return ' '.join(text_lemma)


from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

tfid = TfidfVectorizer(preprocessor=lemmatization_join)

df['content'] = df['content'].apply(lemmatization_join)
df['translated_content'] = df['translated_content'].apply(lemmatization_join)

cv = CountVectorizer()


X = df.drop('problem_domain', axis=1)
X = cv.fit_transform(X['content']).toarray()
y = df['problem_domain']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
RFC = RandomForestClassifier()
GBC = GradientBoostingClassifier()



from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

def evaluate(X_train,X_test,y_train,y_test,model):
    model = model.fit(X_train,y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test,pred)
    con = confusion_matrix(y_test,pred)
    clf_rpt = classification_report(y_test, pred)
    print('confusion matrix',con)
    print(f'{model.__class__.__name__}, --Accuracy-- {acc*100:.2f}%; --Classification Report-- {clf_rpt}')
    return pred

RFC_pred = evaluate(X_train, X_test, y_train, y_test,RFC)
GBC_pred = evaluate(X_train, X_test, y_train, y_test, GBC)
tree_pred = evaluate(X_train, X_test, y_train, y_test, tree)
lr_pred = evaluate(X_train, X_test, y_train, y_test, lr)

from sklearn.model_selection import cross_val_score

def cross_(X,y,model):
    model = model.fit(X,y)
    cv_scores = cross_val_score(model, X,y,cv=10,scoring='accuracy').max()
    print(f'{model.__class__.__name__}, --Cross Validation Scores -- {cv_scores*100:.2f}%')
    return cv_scores

RFC_scores = cross_(X, y, RFC)
GBC_scores = cross_(X,y,GBC)
tree_scores = cross_(X,y,tree)
lr_scores = cross_(X, y, lr)



