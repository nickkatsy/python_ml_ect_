import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.corpus import stopwords


nltk.download('punkt')


df = pd.read_csv('https://raw.githubusercontent.com/nickkatsy/python_ml_ect_/master/arbiter.csv')

df['name'] = df['name'].apply(lambda x: x.lower())
df['line'] = df['line'].apply(lambda x: x.lower())

punctuation_signs = list("?:!.,;")
df['line'] = df['line']

for punct_sign in punctuation_signs:   
    df['line'] = df['line'].str.replace(punct_sign, '')


df['line'] = df['line'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

print(df['line'])


df['line'] = df['line'].apply(lambda x: x.replace('\n', ' '))
df['line'] = df['line'].apply(lambda x: x.replace('\t', ' '))
df['line'] = df['line'].str.replace("    ", " ")
df['line'] = df['line'].str.replace('"', '')



nltk.download('stopwords')
stop_words = list(stopwords.words('english'))
for stop_word in stop_words:
    regex_stopword = r"\b" + stop_word + r"\b"
    df['line'] = df['line'].str.replace(regex_stopword, '')

cv = CountVectorizer()
X = cv.fit_transform(df['line'])
y = df['name']




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,stratify=y,random_state = 42)

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



