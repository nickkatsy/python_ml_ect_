import nltk
from nltk.corpus import stopwords
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:/ML/python/data/arbiter.csv',delimiter=',')
df.isnull().sum()
df.duplicated().sum()



df['name'].drop_duplicates(inplace=True)

fig, axs = plt.subplots(figsize=(6,5)) 
sns.countplot(x='name',data=df,ax=axs)
axs.set_xticklabels(axs.get_xticklabels(),rotation=40,ha="right")
plt.tight_layout()
plt.show()


df['line'] = df['line'].str.lower()
df['name'] =df['name'].str.lower()

import string
PUNC = string.punctuation

def remove_punc(text):
    return text.translate(str.maketrans("","",PUNC))


df['line'] = df['line'].apply(remove_punc)


df['line'] = df['line'].str.replace("\d","")
df['line'] = df['line'].str.replace("[^\w\s]","")


print(df['line'])



from nltk.tokenize import word_tokenize
sw = set(stopwords.words('english'))
nltk.download("omw-1.4")
nltk.download("wordnet")
nltk.download("stopwords")

def remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in sw]
    return ' '.join(filtered_tokens)

df['line'] = df['line'].apply(remove_stopwords)
print(df['line'])


from wordcloud import WordCloud

text = " ".join(i for i in df.line)

wordcloud = WordCloud(
    background_color="#6B5B95",
    colormap="Set2",
    collocations=False).generate(text)

plt.figure(figsize=(12,6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Like Water I EBB")
plt.show()


print(text.count("arbiter"))
print(text.count("prophet"))
print(text.count("council"))


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfid = TfidfVectorizer()


from sklearn.model_selection import train_test_split
X = df['line']
X = cv.fit_transform(X).toarray()
y = df['name']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=1)




from sklearn.naive_bayes import MultinomialNB
MNB = MultinomialNB()



from sklearn.linear_model import PassiveAggressiveClassifier,LogisticRegression
PC = PassiveAggressiveClassifier()
lr = LogisticRegression()

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

def evaluate_model(X_train,X_test,y_train,y_test,model):
    model = model.fit(X_train,y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test,pred)
    clf_rpt = classification_report(y_test,pred)
    print(f'{model.__class__.__name__}, --ACC-- {acc*100:.2f}%; --Clf Rpt-- {clf_rpt}')
    return pred





PC_pred = evaluate_model(X_train, X_test, y_train, y_test, PC)
MNB_pred = evaluate_model(X_train, X_test, y_train, y_test, MNB)
lr_pred = evaluate_model(X_train, X_test, y_train, y_test, lr)



def confusion_matrix_(X_train,X_test,y_train,y_test,model):
    model = model.fit(X_train,y_train)
    pred = model.predict(X_test)
    con = confusion_matrix(y_test,pred)
    heatmap = sns.heatmap(con,annot=True,fmt="d",cmap="Blues")
    heatmap.set_title(f'Confusion Matrix for {model.__class__.__name__}')
    return heatmap

confusion_matrix_(X_train, X_test, y_train, y_test,MNB)
confusion_matrix_(X_train, X_test, y_train, y_test,PC)
confusion_matrix_(X_train, X_test, y_train, y_test,lr)







