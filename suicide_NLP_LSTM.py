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
nltk.download("wordnet")
nltk.download("punkt")


df = pd.read_csv("https://raw.githubusercontent.com/nickkatsy/python_ml_ect_/master/Suicide_Ideation_Dataset(Twitter-based).csv")


df.head(10)

df.isna().sum()
df.dropna(inplace=True)
df.duplicated().sum()
df.drop_duplicates(inplace=True)

df['Suicide'].value_counts().plot(kind='pie',autopct='%1.1f%%')


df['class'] = df['Suicide'].map({"Not Suicide post":0,"Potential Suicide post ":1})

df['class'].value_counts()


PUNC = string.punctuation


def clean_text(text):
    
    
    text = str(text).lower()
    
    text = re.sub(r'<.*?>+', '',text)
    
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    text = re.sub(r'\d', '',text)
    text = re.sub('[%s]' % re.escape(PUNC), '',text)
    
    text = re.sub(' +', ' ', text)
    
    text = re.sub('rt', '',text)

    
    
    return text


df['Tweet'] = df['Tweet'].apply(clean_text)


import contractions

def expand_contractions(text):
    expanded_text = contractions.fix(text)
    return expanded_text

df['Tweet'] = df['Tweet'].apply(expand_contractions)


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


X = df['Tweet']
y = df['class']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.15,random_state=1)


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences,to_categorical

tokenizer = Tokenizer()


tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index
print(len(word_index))



X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


max_length = 0
for sequence in X_train:
    sequence_length = len(sequence)
    if sequence_length > max_length:
        max_length = sequence_length


print(max_length)

X_train = pad_sequences(X_train,max_length,padding='post')
X_test = pad_sequences(X_test,max_length,padding='post')


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,Dense,Bidirectional,Dropout

RNN = Sequential()
RNN.add(Embedding(input_dim=len(word_index)+1, output_dim=150, input_length=max_length))
RNN.add(Bidirectional(LSTM(150, dropout=0.2, recurrent_dropout=0.2)))
RNN.add(Dropout(0.3))
RNN.add(Dense(50,activation='relu'))
RNN.add(Dense(2, activation='sigmoid'))
RNN.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = RNN.fit(X_train,y_train,batch_size=128,epochs=10,validation_data=(X_test, y_test))




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



