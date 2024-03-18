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


model = MultinomialNB()
model.fit(X_train_tfidf, y_train)


y_pred = model.predict(X_test_tfidf)


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


print(classification_report(y_test, y_pred))


unique_characters = data['name'].unique()
for character in unique_characters:
    lines_spoken = get_lines_by_character(character, data)
    print(f"Lines spoken by {character}:")
    print(lines_spoken)
    print(lines_spoken)
