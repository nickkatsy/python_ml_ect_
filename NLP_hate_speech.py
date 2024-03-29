import pandas as pd
import warnings
warnings.filterwarnings('ignore')



df = pd.read_csv('C:/ML/python/data/cyberbullying_tweets.csv',delimiter=',')



df['tweet_text'] = df['tweet_text'].str.lower()

df['tweet_text'] = df['tweet_text'].str.replace('[^\w\s]', '')
df.info()

X = df['tweet_text']
y = df['cyberbullying_type']


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)




from sklearn.feature_extraction.text import TfidfVectorizer

tdift = TfidfVectorizer()

X_train_tfid = tdift.fit_transform(X_train)
X_test_tfid = tdift.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
MNN = MultinomialNB().fit(X_train_tfid,y_train)
MNN_pred = MNN.predict(X_test_tfid)

from sklearn.metrics import classification_report,accuracy_score
clf_rprt_NB = classification_report(y_test, MNN_pred)
print(f'the clf report using Naive Bayes: {clf_rprt_NB}')
acc_NB = accuracy_score(y_test, MNN_pred)
print(f'the accuracy of the Naive Bayes classifier: {acc_NB*100:.2f}%')

#other models


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()


def evaluated_hate_speech(X_train_tfid,X_test_tfid,y_train,y_test,model):
    model = model.fit(X_train_tfid,y_train)
    pred = model.predict(X_test_tfid)
    acc = accuracy_score(y_test, pred)
    clf_rpt = classification_report(y_test,pred)
    print(f'{model.__class__.__name__}, --ACC-- {acc*100:.2f}%; --CLF Report-- {clf_rpt}')
    return pred



lr_pred = evaluated_hate_speech(X_train_tfid, X_test_tfid, y_train, y_test, lr)
tree_pred = evaluated_hate_speech(X_train_tfid, X_test_tfid, y_train, y_test, tree)










