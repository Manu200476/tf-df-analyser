import numpy as np
from newspaper import Article
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_text(url, lang):
    article = Article(url, lenguage=lang)

    article.download()
    article.parse()

    return [article.text.encode('UTF-8')]

def tf_idf_calc(text):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)
    tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(text)
    first_vector_tfidfvectorizer = tfidf_vectorizer_vectors[0]

    data = [
        tfidf_vectorizer.get_feature_names(),
        np.squeeze(np.asarray(first_vector_tfidfvectorizer.T.todense()))
    ]

    tuples = list(zip(*data))

    return tuples

def convert_to_csv(df):
    csv = df.to_csv()
    return csv
