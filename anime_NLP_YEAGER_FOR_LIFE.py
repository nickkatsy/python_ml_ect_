import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('C:/ML/python/data/animequotes.csv', delimiter=',')


df['Character'] = df['Character'].str.lower().str.replace('\W', ' ')
df['Quote'] = df['Quote'].str.lower().str.replace('\W', ' ')
df['Anime'] = df['Anime'].str.lower().str.replace('\W', ' ')

X = df['Quote']
y = df['Character']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42)


tfidf = TfidfVectorizer()
X_tfidf_train = tfidf.fit_transform(X_train)
X_tfidf_test = tfidf.transform(X_test)

from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier,RandomForestClassifier
GBC = GradientBoostingClassifier()
BC = BaggingClassifier()
RFC = RandomForestClassifier()

from sklearn.svm import SVC
svc = SVC(probability=False)
from sklearn.naive_bayes import MultinomialNB
MNB = MultinomialNB()

from sklearn.metrics import accuracy_score, classification_report


def evaluate_model(X_train,X_test,y_train,y_test,model):
    model = model.fit(X_train,y_train)
    pred = model.predict(X_tfidf_test)
    acc = accuracy_score(y_test,pred)
    clf_report = classification_report(y_test,pred)
    print(f'{model.__class__.__name__}, -- Classification Report -- {clf_report} Accuracy {acc*100:.2f}%')
    return pred


BC_pred = evaluate_model(X_tfidf_train, X_tfidf_test, y_train, y_test, BC)
GBC_pred = evaluate_model(X_tfidf_train, X_tfidf_test, y_train, y_test, GBC)
RFC_pred = evaluate_model(X_tfidf_train, X_tfidf_test, y_train, y_test, RFC)
svc_pred = evaluate_model(X_tfidf_train, X_tfidf_test, y_train, y_test, svc)
MNB_pred = evaluate_model(X_tfidf_train, X_tfidf_test, y_train, y_test, MNB)



def get_lines_by_character(character,df):
    lines_spoken = df[df['Character'] == character]['Quote']
    return lines_spoken

unique_characters = df['Character'].unique()
for character in unique_characters:
    print(f'{character}:')
    lines_spoken = get_lines_by_character(character, df)
    print(lines_spoken)


#Yeager is the ultimate
eren_quotes = get_lines_by_character('eren', df)
for quote in eren_quotes:
    print(quote)






