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
    # ... (Kode preprocessing tidak berubah)

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
