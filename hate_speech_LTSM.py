import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
nltk.download("stopwords")
nltk.download("omw-1.4")
nltk.download("wordnet")
nltk.download("punkt")
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('C:/ML/python/data/labeled_data.csv',delimiter=',')

df.dtypes
df.isna().sum()
df.shape
df.drop(['Unnamed: 0','count'],axis=1,inplace=True)
df.duplicated().sum()
df['class'].value_counts()

df['sentiment'] = df['class'].map({0:'Hate_Speech',1:'offensive_language',
                                  2: 'Neither'})


fig, axs = plt.subplots(figsize=(6,5)) 
sns.countplot(x='sentiment',data=df,ax=axs)
axs.set_xticklabels(axs.get_xticklabels(),rotation=40,ha="right")
plt.tight_layout()
plt.show()


df['tweet'] = df['tweet'].str.lower()
print(df['tweet'])





def remove_html_tags(text):
    pattern = r'<.*?>' 
    text = re.sub(pattern, '', text)
    return text

df['tweet'] = df['tweet'].apply(remove_html_tags)

def remove_url(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'',text)


df['tweet'] = df['tweet'].apply(remove_url)




PUNC = string.punctuation

def remove_punctuation(text):
    return text.translate(str.maketrans('','',PUNC))

df['tweet'] = df['tweet'].apply(remove_punctuation)


df['tweet'] = df['tweet'].str.replace("rt","")
df['tweet'] = df['tweet'].str.replace("\d","")
df["tweet"] = df["tweet"].str.replace("[^\w\s]","")
df['tweet'] = df['tweet'].str.replace("  "," ")
df['tweet'] = df['tweet'].str.replace('"','')
df['tweet'] = df['tweet'].str.replace("'s", "")
df['tweet'] = df['tweet'].str.replace("\n","")
df['tweet'] = df['tweet'].str.replace("\t","")


sw = list(stopwords.words('english'))

def remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in sw]
    return ' '.join(filtered_tokens)

df['tweet'] = df['tweet'].apply(remove_stopwords)




lemma = WordNetLemmatizer()

def lemm_text(text):
    tokens = word_tokenize(text)
    lemma_tokens = [lemma.lemmatize(token) for token in tokens]
    return " ".join(lemma_tokens)

df['tweet'] = df['tweet'].apply(lemm_text)




from textblob import TextBlob



text_ = " ".join(i for i in df.tweet)
print(text_)

hateblob = TextBlob(text_)

from wordcloud import WordCloud

text = " ".join(i for i in df.tweet)

wordcloud = WordCloud(
    background_color="#6B5B95",
    colormap="Set2",
    collocations=False).generate(text)

plt.figure(figsize=(10,6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Bad Tweets By Bad People")
plt.show()


#I did not come up with this
# I did not say this
# this is public domain
# I did not say these words
#these people did and shame you them(you possibly)


print(text_.count("bitch"))
print(text_.count("bitches"))
print(text_.count("nigga"))
print(text_.count("niggas"))
print(text_.count("hoe"))
print(text_.count("trash"))
print(text_.count("pussy"))
print(text_.count("fuck"))
print(text_.count("fucking"))
print(text_.count("love"))
print(text_.count("faggot"))


#ok, more uncessary hatred

Hate_tweet = (df['sentiment'] == "Hate_Speech").astype('int32')
Hate_tweet.describe()

offensive_tweets = (df['sentiment'] == "offensive_language").astype('int32')
offensive_tweets.describe()

neither = (df['sentiment'] == "Neither").astype('int32')
neither.value_counts()

plt.figure(figsize=(10,6))
sns.countplot(x=Hate_tweet)
plt.xticks()
plt.show()

sns.countplot(x=offensive_tweets)
plt.show()

sns.countplot(x=neither)
plt.show()


#length of tweets


df['length_of_hate'] = df['tweet'].apply(len)

sns.histplot(x='length_of_hate',hue='sentiment',data=df)
plt.title('Histogram of hateful tweets and just degeneracy')
plt.show()



from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df['hate_speech'] = [1 if X == 1 else 0 for X in df['class']]
from sklearn.model_selection import train_test_split

X = df['tweet']
y = df['hate_speech']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.15,random_state=1)





from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Embedding, Dense,SpatialDropout1D,Dropout





token = Tokenizer()
token.fit_on_texts(X_train)

word_index = token.word_index
print(word_index)


X_train = token.texts_to_sequences(X_train)
X_test = token.texts_to_sequences(X_test)

max_length = 0
for sequence in X_train:
    sequence_length = len(sequence)
    if sequence_length > max_length:
        max_length = sequence_length


print(max_length)

X_train[0]
X_train[123]


from tensorflow.keras.utils import pad_sequences

X_train = pad_sequences(X_train,25,padding="post")
X_test = pad_sequences(X_test,25,padding="post")



RNN = Sequential()
RNN.add(Embedding(len(word_index) + 1,output_dim=100,input_length=25))
RNN.add(SpatialDropout1D(0.5))
RNN.add(LSTM(25, dropout=0.2, recurrent_dropout=0.2))
RNN.add(Dropout(0.5))
RNN.add(Dense(1, activation='sigmoid'))
RNN.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

batch_size = 64

history = RNN.fit(X_train, y_train, epochs=20, batch_size=batch_size,validation_data=(X_test,y_test))

print('accuracy: ',history.history['accuracy'])
print("val_accuracy",history.history['val_accuracy'])
print("val_loss",history.history["val_loss"])
print("loss: ",history.history['loss'])


#bad

#training and val accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


#training and evaluation resluts

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


