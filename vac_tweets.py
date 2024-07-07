import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download("wordnet")
nltk.download("punkt")
nltk.download('omw-1.4')
nltk.download("stopwords")
import re
import string
import demoji



df = pd.read_csv("https://raw.githubusercontent.com/nickkatsy/python_ml_ect_/master/vaccination_tweets.csv")

df.head(10)
df.isna().sum()




def clean_text(text):
    
    text = str(text).lower()

    text = re.sub(r'<.*?>', '',text)
    
    text = re.sub('https?://\S+|www\.\S+', '', text)
    
    text = demoji.replace(text,'')
    
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    
    return text


df['text'] = df['text'].apply(clean_text)



sw = set(stopwords.words("english"))


def remove_stopwords(text):
    tokens = word_tokenize(text)
    cleaned_tokens = [word for word in tokens if word.lower() not in sw]
    return " ".join(cleaned_tokens)

df['text'] = df['text'].apply(remove_stopwords)


lemma = WordNetLemmatizer()

def lemmatizer(text):
    tokens = word_tokenize(text)
    lemma_tokens = [lemma.lemmatize(token) for token in tokens]
    return " ".join(lemma_tokens)    

df['text'] = df['text'].apply(lemmatizer)


from wordcloud import WordCloud
from textblob import TextBlob

text = " ".join(i for i in df['text'])


wc = WordCloud(colormap="Set3",collocations=False).generate(text)
plt.imshow(wc,interpolation="blackman")
plt.axis("off")
plt.show()

blob = TextBlob(text)

def polarity(text):
    return TextBlob(text).polarity

df['polarity'] = df['text'].apply(polarity)


def sentiment(label):
    if label <0:
        return "Negative"
    elif label == 0:
        return "Neutral"
    elif label >= 0:
        return "Positive"


df['senitment'] = df['polarity'].apply(sentiment)


df['senitment'].value_counts().plot(kind='bar',rot=0)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

from sklearn.feature_extraction.text import TfidfVectorizer

tfid = TfidfVectorizer()

X = df['text']
X = tfid.fit_transform(X)
y = df['senitment']
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)


from sklearn.linear_model import LogisticRegression,PassiveAggressiveClassifier
lr = LogisticRegression()
PA = PassiveAggressiveClassifier()


from sklearn.naive_bayes import BernoulliNB,GaussianNB,MultinomialNB

BNB = BernoulliNB()
GNB = GaussianNB()
MNB = MultinomialNB()

from sklearn.metrics import accuracy_score,classification_report

def evaluate_model(X_train,X_test,y_train,y_test,model):
    model = model.fit(X_train,y_train)
    pred = model.predict(X_test)
    acc=  accuracy_score(y_test,pred)
    clf_rpt = classification_report(y_test,pred)
    print(f'{model.__class__.__name__},--Accuracy-- {acc*100:.2f}%; --Classification Report-- {clf_rpt}')
    return pred


lr_pred = evaluate_model(X_train, X_test, y_train, y_test,lr)
PA_pred = evaluate_model(X_train, X_test, y_train, y_test, PA)
BNB_pred = evaluate_model(X_train, X_test, y_train, y_test,BNB)
MNB_pred = evaluate_model(X_train, X_test, y_train, y_test, MNB)



X = df['text']
y = df['senitment']
y = le.fit_transform(y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.15,random_state=1)

from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

word_index = tokenizer.word_index
print(len(word_index))

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)




from tensorflow.keras.utils import pad_sequences,to_categorical

max_length = 0
for sequence in X_train:
    sequence_length = len(sequence)
    if sequence_length > max_length:
        max_length = sequence_length

print("Max Length of Sequences: ",max_length)


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

X_train = pad_sequences(X_train,padding='post')
X_test = pad_sequences(X_test,padding="post")


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Embedding,LSTM,SpatialDropout1D,Dropout

RNN = Sequential()
RNN.add(Embedding(input_dim=(len(word_index)+1),output_dim=100,input_length=max_length))
RNN.add(SpatialDropout1D(0.2))
RNN.add(LSTM(50,dropout=0.1,recurrent_dropout=0.1))
RNN.add(Dropout(0.1))
RNN.add(Dense(50,activation='relu'))
RNN.add(Dropout(0.1))
RNN.add(Dense(3,activation='softmax'))
RNN.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history = RNN.fit(X_train,y_train,epochs=10,batch_size=32,validation_split=0.2)
print(max(history.history['accuracy']))



plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()




plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()




