import numpy as np
from newspaper import Article
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import re
import nltk
nltk.download('stopwords')

stop = list(stopwords.words('spanish'))
def extract_text(url, lang):
    article = Article(url, lenguage=lang)

    article.download()
    article.parse()
    text = article.text.encode().decode()
    text = text.lower()
    text = re.sub("", "", text)
    text = re.sub("(\\d|\\W)+", " ", text)

    return [text]

def tf_idf_calc(text):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, stop_words=set(stop))
    tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(text)
    first_vector_tfidfvectorizer = tfidf_vectorizer_vectors[0]

    data = [
        tfidf_vectorizer.get_feature_names(),
        np.squeeze(np.asarray(np.around(first_vector_tfidfvectorizer.T.todense() * 100, 2)))
    ]

    tuples = dict(zip(*data))
    sort_by_values = dict(sorted(tuples.items(), key=lambda x: x[1], reverse=True))
    first_20_elements = dict(list(sort_by_values.items())[:20])
    tuples = [(k, v) for k, v in first_20_elements.items()]
    return tuples

def convert_to_csv(df):
    csv = df.to_csv()
    return csv
