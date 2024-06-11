import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import re
from textblob import TextBlob
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.stem import WordNetLemmatizer



df = pd.read_csv('C:/ML/python/data/data.csv',delimiter=',')

df.info()
df.isna().sum()
df.duplicated().sum()
df.drop_duplicates(inplace=True)

df['Sentiment'].value_counts()

df['class'] = df['Sentiment'].map({'negative':0,'positive':1,'neutral':2})


fig, axs = plt.subplots(figsize=(6,5)) 
sns.countplot(x='Sentiment',data=df,ax=axs)
axs.set_xticklabels(axs.get_xticklabels(),rotation=40,ha="right")
plt.tight_layout()
plt.show()

df['Sentence'] = df['Sentence'].str.lower()
print(df['Sentence'])


def remove_html_tags(text):
    pattern = r'<.*?>' 
    text = re.sub(pattern, '', text)
    return text

df['Sentence'] = df['Sentence'].apply(remove_html_tags)

def remove_url(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'',text)


df['Sentence'] = df['Sentence'].apply(remove_url)

import string
PUNC = string.punctuation


def remove_punctuations(text):
    return text.translate(str.maketrans("","",PUNC))

df['Sentence'] = df['Sentence'].apply(remove_punctuations)

df['Sentence'] = df['Sentence'].str.replace("\d","")
df['Senetence'] = df['Sentence'].str.replace("[^\w\s]","")
df['Sentence'] = df['Sentence'].str.replace("mn","")

sw = set(stopwords.words("english"))


def remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in sw]
    return ' '.join(filtered_tokens)

df['Sentence'] = df['Sentence'].apply(remove_stopwords)


#lemmatization
lemma = WordNetLemmatizer()

def lemm_text(text):
    tokens = word_tokenize(text)
    lemma_tokens = [lemma.lemmatize(token) for token in tokens]
    return " ".join(lemma_tokens)

df['Sentence'] = df['Sentence'].apply(lemm_text)




text_ = " ".join(word for word in df["Sentence"])

blob = TextBlob(text_)

print(blob.sentences)
print(blob.tags)


wordcloud = WordCloud(colormap='Set2',collocations=False).generate(text_)
plt.imshow(wordcloud,interpolation='mitchell')
plt.show()


df['length'] = df['Sentence'].apply(len)


sns.histplot(x='length',data=df)
plt.title('Longest Sentence')
plt.show()

#sentiment anaylsis

blob_ = []
for i in df['Sentence']:
    blob1 = TextBlob(i).sentiment
    blob_.append(blob1)

blob_df = pd.DataFrame(blob_)

df_blob2 = pd.concat([df.reset_index(drop=True), blob_df], axis=1)
df_blob2.head()

import numpy as np
df_blob2["Sentiment"] =  np.where(df_blob2["polarity"] >= 0 , "Positive", "Negative")



print(df_blob2.value_counts())
df_blob2.groupby('Sentiment').count()



from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


X = df['Sentence']
X = cv.fit_transform(X).toarray()
y = df['Sentiment']
y = le.fit_transform(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)


#naive bayes

from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB

BNB = BernoulliNB()
GNB = GaussianNB()
MNB = MultinomialNB()

from sklearn.linear_model import LogisticRegression,PassiveAggressiveClassifier

lr = LogisticRegression()
PA = PassiveAggressiveClassifier()

from sklearn.metrics import accuracy_score,classification_report

def evaluate_naive_bayes(X_train,X_test,y_train,y_test,model):
    model = model.fit(X_train,y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test,pred)
    clf_rpt = classification_report(y_test, pred)
    print(f'{model.__class__.__name__}, --Accuracy-- {acc*100:.2f}%; --Clf_rpt-- {clf_rpt}')
    return pred


BNB_pred = evaluate_naive_bayes(X_train, X_test, y_train, y_test,BNB)
GNB_pred = evaluate_naive_bayes(X_train, X_test, y_train, y_test, GNB)
MNB_pred = evaluate_naive_bayes(X_train, X_test, y_train, y_test, MNB)
lr_pred = evaluate_naive_bayes(X_train, X_test, y_train, y_test, lr)
PA_pred = evaluate_naive_bayes(X_train, X_test, y_train, y_test, PA)


from sklearn.metrics import confusion_matrix

def confusion_matrix_plot(X_train,X_test,y_train,y_test,model):
    model = model.fit(X_train,y_train)
    pred = model.predict(X_test)
    con = confusion_matrix(y_test,pred)
    heatmap = sns.heatmap(con,annot=True,fmt="d",cmap="coolwarm")
    heatmap.set_title(f'Confusion Matrix for {model.__class__.__name__}')
    return heatmap

confusion_matrix_plot(X_train, X_test, y_train, y_test, BNB)
confusion_matrix_plot(X_train, X_test, y_train, y_test, GNB)
confusion_matrix_plot(X_train, X_test, y_train, y_test,MNB)
confusion_matrix_plot(X_train, X_test, y_train, y_test, lr)
confusion_matrix_plot(X_train, X_test, y_train, y_test, PA)



