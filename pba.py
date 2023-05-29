import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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

st.title("PEMROSESAN BAHASA ALAMI A")
st.write("### Dosen Pengampu : Dr. FIKA HASTARITA RACHMAN, ST., M.Eng")
st.write("#### Kelompok : 5")
st.write("##### Hambali Fitrianto - 200411100074")
st.write("##### Pramudya Dwi Febrianto - 200411100042")
st.write("##### Febrian Achmad Syahputra - 200411100106")

# Navbar
description, preprocessing, implementation = st.columns(3)
dataset = pd.read_csv("https://raw.githubusercontent.com/Feb11F/dataset/main/dieng_sentiment_pn.csv")

# Data Set Description
with description:
    st.subheader("Description")
    # ... (Kode deskripsi tidak berubah)

with preprocessing:
    st.subheader("Preprocessing Data")

    dataset['ulasan'] = dataset['ulasan'].str.lower()
    st.write("Case Folding Result:")
    st.write(dataset['ulasan'].head())

    st.write("Tokenize:")
    def hapus_tweet_khusus(text):
        text = text.replace('\\t', " ").replace('\\n', " ").replace('\\u', " ").replace('\\', "")
        text = text.encode('ascii', 'replace').decode('ascii')
        text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)", " ", text).split())
        return text.replace("http://", " ").replace("https://", " ")
    
    dataset['ulasan'] = dataset['ulasan'].apply(hapus_tweet_khusus)
    
    def hapus_nomor(text):
        return re.sub(r"\d+", "", text)
    
    dataset['ulasan'] = dataset['ulasan'].apply(hapus_nomor)
    
    def hapus_tanda_baca(text):
        return text.translate(str.maketrans("", "", string.punctuation))
    
    dataset['ulasan'] = dataset['ulasan'].apply(hapus_tanda_baca)
    
    def hapus_whitespace_LT(text):
        return text.strip()
    
    dataset['ulasan'] = dataset['ulasan'].apply(hapus_whitespace_LT)
    
    def hapus_whitespace_multiple(text):
        return re.sub('\s+', ' ', text)
    
    dataset['ulasan'] = dataset['ulasan'].apply(hapus_whitespace_multiple)
    
    def hapus_single_char(text):
        return re.sub(r"\b[a-zA-Z]\b", "", text)
    
    dataset['ulasan'] = dataset['ulasan'].apply(hapus_single_char)
    
    def word_tokenize_wrapper(text):
        tokenizer = RegexpTokenizer(r'dataran\s+tinggi|jawa\s+tengah|[\w\']+')
        tokens = tokenizer.tokenize(text)
        return tokens
    
    dataset['ulasan_tokens'] = dataset['ulasan'].apply(word_tokenize_wrapper)
    st.write(dataset['ulasan_tokens'].head())
    
    def freqDist_wrapper(text):
        return FreqDist(text)
    
    dataset['ulasan_tokens_fdist'] = dataset['ulasan_tokens'].apply(freqDist_wrapper)
    st.write(dataset['ulasan_tokens_fdist'].head())
    
    st.write("Filtering (Stopword Removal):")
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

    dataset['ulasan_tokens_WSW'] = dataset['ulasan_tokens'].apply(stopwords_removal)
    st.write(dataset['ulasan_tokens_WSW'].head())

    st.write("Stemming:")
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    def stemmed_wrapper(term):
        return stemmer.stem(term)

    term_dict = {}

    for document in dataset['ulasan_tokens_WSW']:
        for term in document:
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

    dataset['ulasan_tokens_stemmed'] = dataset['ulasan_tokens_WSW'].apply(get_stemmed_term)
    st.write(dataset['ulasan_tokens_stemmed'].head())


with implementation:
    st.subheader("Implementation")
    # ... (Kode implementasi lainnya)

    # Penambahan tahap processing saat ada inputan ulasan
    user_input = st.text_input("Masukkan ulasan:")
    if user_input:
        user_input = user_input.lower()
        user_input = hapus_tweet_khusus(user_input)
        user_input = hapus_nomor(user_input)
        user_input = hapus_tanda_baca(user_input)
        user_input = hapus_whitespace_LT(user_input)
        user_input = hapus_whitespace_multiple(user_input)
        user_input = hapus_single_char(user_input)
        user_input_tokens = word_tokenize_wrapper(user_input)
        user_input_tokens_WSW = stopwords_removal(user_input_tokens)
        user_input_tokens_stemmed = get_stemmed_term(user_input_tokens_WSW)

        st.write("Hasil preprocessing inputan ulasan:")
        st.write(user_input_tokens_stemmed)

    # Menampilkan data set
    st.subheader("Data Set")
    st.dataframe(dataset)  # Tampilkan dataset menggunakan st.dataframe()
