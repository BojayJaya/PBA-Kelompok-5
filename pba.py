import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import nltk
nltk.download('punkt')
import string 
import re
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
nltk.download('stopwords')
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pickle
import ast
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score



ulasan = st.text_area('Masukkan kata yang akan di analisa :')
submit = st.button("submit")

if submit:
    ulasan = ulasan.lower()
    def hapus_tweet_khusus(text):
        text = text.replace('\\t', " ").replace('\\n', " ").replace('\\u', " ").replace('\\', "")
        text = text.encode('ascii', 'replace').decode('ascii')
        text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)", " ", text).split())
        return text.replace("http://", " ").replace("https://", " ")

    ulasan = hapus_tweet_khusus(ulasan)

    def hapus_nomor(text):
        return re.sub(r"\d+", "", text)

    ulasan = hapus_nomor(ulasan)

    def hapus_tanda_baca(text):
        return text.translate(str.maketrans("", "", string.punctuation))

    ulasan = hapus_tanda_baca(ulasan)

    def hapus_whitespace_LT(text):
        return text.strip()

    ulasan = hapus_whitespace_LT(ulasan)

    def hapus_whitespace_multiple(text):
        return re.sub('\s+', ' ', text)

    ulasan = hapus_whitespace_multiple(ulasan)

    def hapus_single_char(text):
        return re.sub(r"\b[a-zA-Z]\b", "", text)

    ulasan = hapus_single_char(ulasan)

    def word_tokenize_wrapper(text):
        tokenizer = RegexpTokenizer(r'dataran\s+tinggi|jawa\s+tengah|[\w\']+')
        tokens = tokenizer.tokenize(text)
        return tokens

    ulasan = word_tokenize_wrapper(ulasan)

    def freqDist_wrapper(text):
        return FreqDist(text)

    ulasan = freqDist_wrapper(ulasan)

    list_stopwords = stopwords.words('indonesian')
    list_stopwords.extend(["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 'kalo', 'amp', 'biar', 'bikin', 'bilang',
                            'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 'si', 'tau', 'tdk', 'tuh', 'utk', 'ya',
                            'jd', 'jgn', 'sdh', 'aja', 'n', 't', 'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                            '&amp', 'yah'])

    txt_stopword = pd.read_csv("https://raw.githubusercontent.com/masdevid/ID-Stopwords/master/id.stopwords.02.01.2016.txt", names=["stopwords"], header=None)

    list_stopwords.extend(txt_stopword["stopwords"][0].split(' '))
    list_stopwords = set(list_stopwords)

    def stopwords_removal(words):
        return [word for word in words if word not in list_stopwords]

    ulasan = stopwords_removal(ulasan)

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    def stemmed_wrapper(term):
        return stemmer.stem(term)

    term_dict = {}

    for term in ulasan:
        if term not in term_dict:
            term_dict[term] = ' '

    st.write(len(term_dict))
    st.write("------------------------")

    for term in term_dict:
        term_dict[term] = stemmed_wrapper(term)
        st.write(term, ":", term_dict[term])

    st.write(term_dict)
    st.write("------------------------")

    def get_stemmed_term(document):
        return [term_dict[term] for term in document]

    ulasan = get_stemmed_term(ulasan)
    
    
    Data_ulasan = pd.read_csv("https://raw.githubusercontent.com/BojayJaya/PBA-Kelompok-5/main/hasil_preprocessing.csv")
    Data_ulasan = pd.DataFrame(Data_ulasan)
    st.write(Data_ulasan)
    
    ulasan = Data_ulasan['ulasan_hasil_preprocessing']
    sentimen = Data_ulasan['label']
    X_train, X_test, y_train, y_test = train_test_split(ulasan, sentimen, test_size=0.2, random_state=42)
    
    st.write(X_train)
