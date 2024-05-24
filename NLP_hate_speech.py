import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.corpus import stopwords


nltk.download('punkt')

df = pd.read_csv('https://raw.githubusercontent.com/nickkatsy/python_ml_ect_/master/cyberbullying_tweets.csv')
df.info()


df['tweet_text'] = df['tweet_text'].apply(lambda x: x.lower())



punctuation_signs = list("?:!.,;")
df['tweet_text'] = df['tweet_text']

for punct_sign in punctuation_signs:   
    df['tweet_text'] = df['tweet_text'].str.replace(punct_sign, '')

df['tweet_text'] = df['tweet_text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

print(df['tweet_text'])

df['tweet_text'] = df['tweet_text'].apply(lambda x: x.replace('\n', ' '))
df['tweet_text'] = df['tweet_text'].apply(lambda x: x.replace('\t', ' '))
df['tweet_text'] = df['tweet_text'].str.replace("    ", " ")
df['tweet_text'] = df['tweet_text'].str.replace('"', '')


nltk.download('stopwords')
stop_words = list(stopwords.words('english'))
for stop_word in stop_words:
    regex_stopword = r"\b" + stop_word + r"\b"
    df['tweet_text'] = df['tweet_text'].str.replace(regex_stopword, '')


cv = CountVectorizer(max_features = 75)
X = cv.fit_transform(df['tweet_text']).toarray()
y = df['cyberbullying_type']




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y, random_state = 42)

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

