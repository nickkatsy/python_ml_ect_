import pandas as pd
import warnings
warnings.filterwarnings('ignore')



df = pd.read_csv('https://raw.githubusercontent.com/nickkatsy/python_ml_ect_/master/cyberbullying_tweets.csv')
df.info()

df['tweet_text'] = df['tweet_text'].str.lower()
df['tweet_text'] = df['tweet_text'].str.replace('W', '')





from sklearn.feature_extraction.text import TfidfVectorizer

tfid = TfidfVectorizer()

X = df['tweet_text']
y = df['cyberbullying_type']

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)

X_train = tfid.fit_transform(X_train)
X_test = tfid.transform(X_test)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()

from sklearn.naive_bayes import MultinomialNB
MNB = MultinomialNB()

from sklearn.metrics import accuracy_score,classification_report


def evaluate_model(X_train,X_test,y_train,y_test,model):
    model = model.fit(X_train,y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test,pred)
    clf_rpt = classification_report(y_test,pred)
    print(f'{model.__class__.__name__}, -Accuracy- {acc*100:.2f}%; --clf_rpt-- {clf_rpt}')
    return pred


tree_pred = evaluate_model(X_train, X_test, y_train, y_test, tree)
lr_pred = evaluate_model(X_train, X_test, y_train, y_test, lr)
MNB_pred = evaluate_model(X_train, X_test, y_train, y_test, MNB)

