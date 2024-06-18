import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import string
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')
nltk.download("stopwords")
nltk.download("wornet")
nltk.download("punkt")


df = pd.read_csv("C:/ML/python/data/Suicide_Ideation_Dataset(Twitter-based).csv",delimiter=',')


df.head(10)

df.isna().sum()
df.dropna(inplace=True)
df.duplicated().sum()
df.drop_duplicates(inplace=True)

df['Suicide'].value_counts().plot(kind='pie',autopct='%1.1f%%')


df['class'] = df['Suicide'].map({"Not Suicide post":0,"Potential Suicide post ":1})

df['class'].value_counts()

df['Tweet'] = df['Tweet'].str.lower()


def remove_html_tags(text):
    pattern = r'<.*?>'
    text = re.sub(pattern, "", text)
    return text


df['Tweet'] = df['Tweet'].str.lower()

df['Tweet'] = df['Tweet'].str.replace("\d","")
df['Tweet'] = df['Tweet'].str.replace("[^\w\s]","")
df['Tweet'] = df['Tweet'].str.replace("rt","")




def remove_url(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'',text)

df['Tweet'] = df['Tweet'].apply(remove_url)



PUNC = string.punctuation


def remove_punctuations(text):
    return text.translate(str.maketrans("","",PUNC))


df['Tweet'] = df['Tweet'].apply(remove_punctuations)

print(df['Tweet'])


sw = set(stopwords.words("english"))


def remove_stopwords(text):
    tokens = word_tokenize(text)
    cleaned_tokens = [word for word in tokens if word.lower() not in sw]
    return " ".join(cleaned_tokens)

df['Tweet'] = df['Tweet'].apply(remove_stopwords)



lemmatizer = WordNetLemmatizer()

def lemmatization(text):
    tokens = word_tokenize(text)
    lemma_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(lemma_tokens)

df['Tweet'] = df['Tweet'].apply(lemmatization)

text = " ".join(i for i in df['Tweet'])


from wordcloud import WordCloud
import matplotlib.pyplot as plt


wc = WordCloud(colormap='Set2',collocations=False).generate(text)
plt.imshow(wc,interpolation='blackman')
plt.show()


from textblob import TextBlob
blob = TextBlob(text)

print(blob.word_counts)
print(blob.sentiment_assessments)


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

X = df['Tweet']
X = cv.fit_transform(X).toarray()
y = df['class']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)

from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB

MNB = MultinomialNB()
GNB = GaussianNB()
BNB = BernoulliNB()

from sklearn.linear_model import LogisticRegression,PassiveAggressiveClassifier
lr = LogisticRegression()
PA = PassiveAggressiveClassifier()



from sklearn.metrics import accuracy_score,classification_report

def evaluate_bayes(X_train,X_test,y_train,y_test,model):
    model = model.fit(X_train,y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test,pred)
    clf_rpt = classification_report(y_test,pred)
    print(f'{model.__class__.__name__}, --Accuracy-- {acc*100:.2f}%; --Classification Report-- {clf_rpt}')
    return pred


MNB_pred = evaluate_bayes(X_train, X_test, y_train, y_test, MNB)
GNB_pred = evaluate_bayes(X_train, X_test, y_train, y_test, GNB)
BNB_pred = evaluate_bayes(X_train, X_test, y_train, y_test, BNB)
PA_pred = evaluate_bayes(X_train, X_test, y_train, y_test, PA)
lr_pred = evaluate_bayes(X_train, X_test, y_train, y_test, lr)




