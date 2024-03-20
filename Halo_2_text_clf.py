import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


def get_lines_by_character(character_name, data):

    lines_spoken = data[data['name'] == character_name]['line']
    return lines_spoken

data = pd.read_csv('https://raw.githubusercontent.com/nickkatsy/python_ml_ect_/master/arbiter.csv')

data['line'] = data['line'].str.lower()
data['line'] = data['line'].str.replace('[^\w\s]', '')

X = data['line']
y = data['name']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


from sklearn.ensemble import GradientBoostingClassifier
GBC = GradientBoostingClassifier()
from sklearn.svm import SVC

GBC = GradientBoostingClassifier()
MNM = MultinomialNB()
svc = SVC(probability=True)

def evaluate(X_train_tfidf, X_test_tfidf, y_train, y_test, model):
    model = model.fit(X_train_tfidf, y_train)
    pred = model.predict(X_test_tfidf)
    clf_report = classification_report(y_test, pred)
    acc = accuracy_score(y_test, pred)
    print(f'{model.__class__.__name__}, --Classification Report--\n{clf_report}')
    print(f'{model.__class__.__name__}, --Accuracy-- {acc*100:.2f}%')
    return pred

GBC_pred = evaluate(X_train_tfidf, X_test_tfidf, y_train, y_test, GBC)
MNM_pred = evaluate(X_train_tfidf, X_test_tfidf, y_train, y_test, MNM)
svc_pred = evaluate(X_train_tfidf, X_test_tfidf, y_train, y_test, svc)







unique_characters = data['name'].unique()
for character in unique_characters:
    lines_spoken = get_lines_by_character(character, data)
    print(f"Lines spoken by {character}:")
    print(lines_spoken)
