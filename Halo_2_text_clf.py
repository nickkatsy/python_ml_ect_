import pandas as pd
import warnings
warnings.filterwarnings('ignore')



df = pd.read_csv('https://raw.githubusercontent.com/nickkatsy/python_ml_ect_/master/arbiter.csv')

df['name'] = df['name'].str.replace('W','')
df['line'] = df['line'].str.replace('W','')



from sklearn.feature_extraction.text import TfidfVectorizer

tfid = TfidfVectorizer()


X = df['line']
y = df['name']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)


X_train = tfid.fit_transform(X_train)
X_test = tfid.transform(X_test)



from sklearn.naive_bayes import MultinomialNB

MNB = MultinomialNB()



from sklearn.ensemble import GradientBoostingClassifier,BaggingClassifier

GBC = GradientBoostingClassifier()
BC = BaggingClassifier()

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


from sklearn.metrics import accuracy_score,classification_report


def evaluate_model(X_train,X_test,y_train,y_test,model):
    model = model.fit(X_train,y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test,pred)
    clf_rpt = classification_report(y_test,pred)
    print(f'{model.__class__.__name__}, --ACC-- {acc*100:.2f}%; -- CLF RPT -- {clf_rpt}')
    return pred

lr_pred = evaluate_model(X_train, X_test, y_train, y_test, lr)
MNB_pred = evaluate_model(X_train, X_test, y_train, y_test, MNB)
BC_pred = evaluate_model(X_train, X_test, y_train, y_test,BC)
GBC_pred = evaluate_model(X_train, X_test, y_train, y_test, GBC)



def get_lines_by_character(character_name,df):

    lines_spoken = df[df['name'] == character_name]['line']
    return lines_spoken


unique_characters = df['name'].unique()
for character in unique_characters:
    lines_spoken = get_lines_by_character(character, df)
    print(f"Lines spoken by {character}:")
    print(lines_spoken)
