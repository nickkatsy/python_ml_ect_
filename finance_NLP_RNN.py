import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.stem import WordNetLemmatizer
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("wordnet")
import pandas as pd
import warnings
warnings.filterwarnings("ignore")



df = pd.read_csv("C:/ML/python/data/data.csv",delimiter=',')
df.head(10)
df.dtypes

df.duplicated().sum()
df.drop_duplicates(inplace=True)
df.isna().sum()

df['Sentiment'].value_counts().plot(kind='pie',autopct="%1.1f%%")


sns.histplot(df['Sentiment'])
PUNC = string.punctuation


def clean_text(text):
    
    
    text = str(text).lower()
    
    text = re.sub(r'<.*?>+', '',text)
    
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    text = re.sub(r'\d', '',text)
    text = re.sub('[%s]' % re.escape(PUNC), '',text)
    
    text = re.sub(' +', ' ', text)
    
    text = re.sub(r'esi', '',text)
    text = re.sub(r'afx', '',text)

    text = re.sub(r'bk', '',text)
    
    text = re.sub(r'kci', '',text)

    
    
    return text




df['Sentence'] = df['Sentence'].apply(clean_text)

df['Sentence'] = df['Sentence'].str.replace('mn', '')



import contractions

def expand_contractions(text):
    expanded_text = contractions.fix(text)
    return expanded_text


df['Sentence'] = df['Sentence'].apply(expand_contractions)


sw = set(stopwords.words("english"))


def remove_stopwords(text):
    tokens = word_tokenize(text)
    cleaned_tokens = [word for word in tokens if word.lower() not in sw]
    return " ".join(cleaned_tokens)

df['Sentence'] = df['Sentence'].apply(remove_stopwords)


lemma = WordNetLemmatizer()


def lemm_text(text):
    tokens = word_tokenize(text)
    lemma_tokens = [lemma.lemmatize(token) for token in tokens]
    return " ".join(lemma_tokens)



df['Sentence'] = df['Sentence'].apply(lemm_text)



from wordcloud import WordCloud
from textblob import TextBlob


text = " ".join(word for word in df['Sentence'])


blob = TextBlob(text)



#wordcloud

wc = WordCloud(colormap='Set2',collocations=False).generate(text)
plt.imshow(wc,interpolation='spline16')
plt.show()




from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

X = df['Sentence']
X = cv.fit_transform(X).toarray()
y = df['Sentiment']
y = y.map({"negative":0,'positive':1,'neutral':2})
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)


from sklearn.linear_model import PassiveAggressiveClassifier,LogisticRegression
lr = LogisticRegression()
PA = PassiveAggressiveClassifier()


from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
BNB = BernoulliNB()
GNB = GaussianNB()
MNB = MultinomialNB()

from sklearn.metrics import accuracy_score,classification_report

def evaluate_model(X_train,X_test,y_train,y_test,model):
    model = model.fit(X_train,y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test,pred)
    clf_rpt = classification_report(y_test,pred)
    print(f'{model.__class__.__name__}, --Accuracy-- {acc*100:.2f}%; --Classification Report-- {clf_rpt}')
    return pred


lr_pred = evaluate_model(X_train, X_test, y_train, y_test, lr)
PA_pred = evaluate_model(X_train, X_test, y_train, y_test,PA)
BNB_pred = evaluate_model(X_train, X_test, y_train, y_test,BNB)
GNB_pred = evaluate_model(X_train, X_test, y_train, y_test, GNB)
MNB_pred = evaluate_model(X_train, X_test, y_train, y_test, MNB)



##train/test split for RNN

X = df['Sentence']
y = df['Sentiment']
y = le.fit_transform(y)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,Dense,Bidirectional,Dropout,SpatialDropout1D
from tensorflow.keras.utils import to_categorical,pad_sequences
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.15,random_state=1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

##
tokenizer = Tokenizer()

tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index
print("length of word index: ",len(word_index))

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)


max_length = 0
for sequence in X_train:
    sequence_length = len(sequence)
    if sequence_length > max_length:
        max_length = sequence_length


print("max length:",max_length)


X_train = pad_sequences(X_train,max_length,padding='post')
X_test = pad_sequences(X_test,max_length,padding='post')

RNN = Sequential()
RNN.add(Embedding(input_dim=len(word_index)+1,output_dim=50,input_length=max_length))
RNN.add(SpatialDropout1D(0.1))
RNN.add(Bidirectional(LSTM(15,dropout=0.1,recurrent_dropout=0.1)))
RNN.add(Dropout(0.2))
RNN.add(Dense(50,activation='relu'))
RNN.add(Dropout(0.3))
RNN.add(Dense(3,activation='sigmoid'))
RNN.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history = RNN.fit(X_train,y_train,batch_size=64,epochs=10,validation_data=(X_test,y_test))



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



