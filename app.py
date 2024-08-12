import streamlit as st
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

news_df = pd.read_csv('data.csv')
news_df = news_df.fillna(' ')
news_df.drop(columns=['Unnamed: 0'],inplace=True)
X = news_df.drop('label', axis=1)
y = news_df['label']
stop_words=set(stopwords.words('english'))
def clean_text(text):
    ps = PorterStemmer()
    
    text = re.sub("[^a-zA-Z]", ' ', text)
    text = text.lower()
    text = text.split()
    
    text = [ps.stem(word) for word in text if word not in stop_words]
    text=' '.join(text)
    
    return text

news_df['text']=news_df['text'].apply(lambda x:clean_text(x))





# Vectorize data
X = news_df['text'].values
y = news_df['label'].values
vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

model = LogisticRegression()
model.fit(X_train,Y_train)


st.title('Fake News Detector')
input_text = st.text_input('Enter news Article')

def prediction(input_text):
    input_data = vector.transform([input_text])
    prediction = model.predict(input_data)
    return prediction[0]

if input_text:
    pred = prediction(input_text)
    if pred == 1:
        st.write('The News is Fake')
    else:
        st.write('The News Is Real')