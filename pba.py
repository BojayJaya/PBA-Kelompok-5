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

st.title("PEMROSESAN BAHASA ALAMI A")
st.write("### Dosen Pengampu : Dr. FIKA HASTARITA RACHMAN, ST., M.Eng")
st.write("#### Kelompok : 5")
st.write("##### Hambali Fitrianto - 200411100074")
st.write("##### Pramudya Dwi Febrianto - 200411100042")
st.write("##### Febrian Achmad Syahputra - 200411100106")

#Navbar
ekstraksi_fitur, implementation = st.tabs(["Ekstraksi Fitur", "Implementation"])
dataset = pd.read_csv("https://raw.githubusercontent.com/Feb11F/dataset/main/dieng_sentiment_pn.csv")

with ekstraksi_fitur:
    st.write("Menyimpan data hasil preprocessing ke pickle")

# Implementasi dengan Streamlit
with implementation:
    st.subheader("Preprocessing Data")

    # Fungsi untuk preprocessing ulasan
    def preprocess_text(text):
        text = text.lower()
        text = hapus_tweet_khusus(text)
        text = hapus_nomor(text)
        text = hapus_tanda_baca(text)
        text = hapus_whitespace_LT(text)
        text = hapus_whitespace_multiple(text)
        text = hapus_single_char(text)
        tokens = word_tokenize_wrapper(text)
        tokens = stopwords_removal(tokens)
        stemmed_tokens = get_stemmed_term(tokens)
        return stemmed_tokens

    st.write("Masukkan ulasan di bawah ini:")
    input_text = st.text_input("Silahkan Masukkan Ulasan Anda :")

    if st.button("Prediksi"):
        # Preprocessing ulasan input
        processed_text = preprocess_text(input_text)

        # Mengubah input ulasan menjadi vektor
        input_vector = text_to_vector(' '.join(processed_text), tfidf_dict)
        input_vector = np.array(input_vector).reshape(1, -1)

        # Melakukan prediksi pada input ulasan
        predicted_label = knn_classifier.predict(input_vector)

        # Menampilkan hasil prediksi
        st.write("Hasil Prediksi:")
        st.write(f"Ulasan: {input_text}")
        st.write(f"Label: {predicted_label[0]}")

    # Melakukan ekstraksi fitur pada data uji
    X_test = dataset['ulasan']
    y_test = dataset['label']
    
    # Fungsi untuk mengubah teks menjadi vektor fitur menggunakan tfidf_dict
    def extract_features(text, tfidf_dict):
        features = []
        for sentence in text:
            tokens = preprocess_text(sentence)
            vector = text_to_vector(' '.join(tokens), tfidf_dict)
            features.append(vector)
        return features


    # Melakukan ekstraksi fitur pada data uji
    X_test = dataset['ulasan']
    y_test = dataset['label']
    X_test_vectors = extract_features(X_test, tfidf_dict)

