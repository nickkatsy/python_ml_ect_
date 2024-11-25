import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch.utils.data import DataLoader
import nltk
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")
nltk.download("wordnet")



df = pd.read_csv("C:/ML/python/data/DATASET.csv",delimiter=',')
df.info()
df.nunique()
df['label'].value_counts().plot(kind='bar',rot=0)
plt.show()


df.isna().sum()
df['Review'] = df['Review'].fillna("")




df.duplicated().sum()
df.drop_duplicates(inplace=True)


df['Review'].head(10)


df['Review'] = df['Review'].str.lower()

import demoji
import string
import re


def clean_text(text):
    
    
    text = re.sub(r'<.*?>', '',text)
    
    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = demoji.replace(text,'')

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    
    text = re.sub(r'\@w+|\#','',text)
    

    
    
    return text

df['Review'] = df['Review'].apply(clean_text)


df['Review'].head(10)


sw = stopwords.words("english")

def remove_stopwords(text):
    tokens = word_tokenize(text)
    cleaned_tokens = [word for word in tokens if word.lower() not in sw]
    return " ".join(cleaned_tokens)


df['Review'] = df['Review'].apply(remove_stopwords)


lemma = WordNetLemmatizer()

def lemmatizer(text):
    tokens = word_tokenize(text)
    lemma_tokens = [lemma.lemmatize(token) for token in tokens]
    return " ".join(lemma_tokens)


df['Review'] = df['Review'].apply(lemmatizer)


review_text = " ".join(i for i in df['Review'])



from wordcloud import WordCloud
from textblob import TextBlob

wc = WordCloud(colormap="Set2",collocations=False).generate(review_text)
plt.imshow(wc,interpolation="bilinear")
plt.axis("off")
plt.show()


blob = TextBlob(review_text).words

from nltk.probability import FreqDist

most_common_words = FreqDist(blob).most_common(50)
print(most_common_words)



def polarity(text):
    return TextBlob(text).polarity



df['polarity'] = df['Review'].apply(polarity)



def sentiment(label):
    if label <0:
        return "Negative"
    elif label == 0:
        return "Neutral"
    elif label>0:
        return "Positive"



df['label'] = [1 if X == "POSITIVE" else 0 for X in df['label']]


df['sentiment'] = df['polarity'].apply(sentiment)

df['sentiment'].value_counts().plot(kind='pie',autopct='%1.1f%%')
plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

tokens = tokenizer.tokenize(review_text)
print(tokenizer.sep_token,tokenizer.sep_token_id)
tokenizer.cls_token,tokenizer.cls_token_id
tokenizer.all_special_tokens



token_lens = []

for txt in df.Review:
    tokens = tokenizer.encode(txt, max_length=512)
    token_lens.append(len(tokens))


sns.distplot(token_lens)
plt.xlim([0, 256]);
plt.xlabel('Token count')
len(token_lens)



MAX_LEN = 512
df.dtypes

class spotify_dataset:
    def __init__(self,Review,targets,tokenizer,max_len):
        self.Review = Review
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.Review)


    def __getitem__(self,item):
        Review = str(self.Review[item])
        target = self.targets[item]
        
        
        encoding = self.tokenizer.encode_plus(
            Review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt",
            )
        
        return{
            "Review_text":Review,
            "input_ids":encoding['input_ids'].flatten(),
            "attention_mask":encoding['attention_mask'].flatten(),
            "targets":torch.tensor(target,dtype=torch.long)
            }
    
    



from sklearn.model_selection import train_test_split

df_train,df_test = train_test_split(df,test_size=0.2, random_state=0)
df_train,df_val = train_test_split(df_test,test_size=.5,random_state=0)


BATCH_SIZE = 5

df.dtypes

def get_dataloader(df,tokenizer,max_len,batch_size):
    ds = spotify_dataset(
        Review = df['Review'].to_list(),
        targets = df['label'].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len,
        )
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        num_workers=0
        )




train_dataloader = get_dataloader(df_train, tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE)
test_dataloader = get_dataloader(df_test, tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE)
val_dataloader = get_dataloader(df_val, tokenizer, max_len=MAX_LEN, batch_size=BATCH_SIZE)
            

data = next(iter(train_dataloader))
print(data.keys())

print(data['input_ids'].shape)
print(data['attention_mask'].shape)
print(data['targets'].shape)


BERT_MODEL = BertModel.from_pretrained(MODEL_NAME,return_dict=True)




class Bertclassifier(nn.Module):
    

    def __init__(self, n_classes):
        super(Bertclassifier, self).__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask
        )

        output = self.drop(pooled_output)
        return self.out(output)



model = Bertclassifier(n_classes=2)
model = model.to(device)




print(BERT_MODEL.config.hidden_size)


EPOCHS = 5


optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)

total_steps = len(train_dataloader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

 
loss_fn = nn.CrossEntropyLoss().to(device)



def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        

        loss.backward()
        

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    return correct_predictions.double() / n_examples, np.mean(losses)







def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
            
    return correct_predictions.double() / n_examples, np.mean(losses)




history = defaultdict(list)
best_accuracy = 0

for epoch in range(EPOCHS):
    

    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("-" * 5)
    
    train_acc, train_loss = train_epoch(
        model,
        train_dataloader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(df_train)
    )
    
    print(f"Train loss {train_loss} accuracy {train_acc}")
    

    val_acc, val_loss = eval_model(
        model,
        val_dataloader,
        loss_fn,
        device,
        len(df_val)
    )
    
    print(f"Val   loss {val_loss} accuracy {val_acc}")
    print()
    
    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)
    

    if val_acc > best_accuracy:
        torch.save(model.state_dict(), 'bert_model.bin')
        best_accuracy = val_acc










