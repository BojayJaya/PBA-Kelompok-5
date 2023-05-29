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

st.title("PEMROSESAN BAHASA ALAMI A")
st.write("### Dosen Pengampu : Dr. FIKA HASTARITA RACHMAN, ST., M.Eng")
st.write("#### Kelompok : 5")
st.write("##### Hambali Fitrianto - 200411100074")
st.write("##### Pramudya Dwi Febrianto - 200411100042")
st.write("##### Febrian Achmad Syahputra - 200411100106")

#Navbar
ekstraksi_fitur, implementation = st.tabs([, "Ekstraksi Fitur", "Implementation"])
dataset = pd.read_csv("https://raw.githubusercontent.com/Feb11F/dataset/main/dieng_sentiment_pn.csv")

with ekstraksi_fitur:
    st.write("Menyimpan data hasil preprocessing ke pickle")
    with open('data.pickle', 'wb') as file:
        pickle.dump(dataset, file)

    # Memuat data dari file pickle
    with open('data.pickle', 'rb') as file:
        loaded_data = pickle.load(file)

    Data_ulasan = pd.DataFrame(loaded_data, columns=["label", "ulasan"])
    Data_ulasan.head()

    ulasan = Data_ulasan['ulasan']
    sentimen = Data_ulasan['label']
    X_train, X_test, y_train, y_test = train_test_split(ulasan, sentimen, test_size=0.2, random_state=42)

    def convert_text_list(texts):
        try:
            texts = ast.literal_eval(texts)
            if isinstance(texts, list):
                return texts
            else:
                return []
        except (SyntaxError, ValueError):
            return []

    Data_ulasan["ulasan_list"] = Data_ulasan["ulasan"].apply(convert_text_list)
    st.write(Data_ulasan["ulasan_list"][90])
    st.write("\ntype: ", type(Data_ulasan["ulasan_list"][90]))

    # Ekstraksi fitur menggunakan TF-IDF
    def calculate_tf(corpus):
        tf_dict = {}
        for document in corpus:
            words = document.split()
            for word in words:
                if word not in tf_dict:
                    tf_dict[word] = 1
                else:
                    tf_dict[word] += 1
        total_words = sum(tf_dict.values())
        for word in tf_dict:
            tf_dict[word] = tf_dict[word] / total_words
        return tf_dict
        st.write(tf_dict)

    def calculate_df(corpus):
        df_dict = {}
        for document in corpus:
            words = set(document.split())
            for word in words:
                if word not in df_dict:
                    df_dict[word] = 1
                else:
                    df_dict[word] += 1
        return df_dict

    def calculate_idf(corpus):
        idf_dict = {}
        N = len(corpus)
        df_dict = calculate_df(corpus)
        for word in df_dict:
            idf_dict[word] = np.log(N / df_dict[word])
        return idf_dict

    def calculate_tfidf(tf_dict, idf_dict):
        tfidf_dict = {}
        for word in tf_dict:
            if word in idf_dict:
                tfidf_dict[word] = tf_dict[word] * idf_dict[word]
            else:
                tfidf_dict[word] = 0
        return tfidf_dict

    tf_train = calculate_tfidf(calculate_tf(X_train), calculate_idf(X_train))
    tf_test = calculate_tfidf(calculate_tf(X_test), calculate_idf(X_train))

    for i, document in enumerate(X_train):
        tfidf_dict = calculate_tfidf(calculate_tf([document]), calculate_idf(X_train))
        st.write(f"Document {i+1}:")
        for word, tfidf in tfidf_dict.items():
            st.write(f"{word}: {tfidf}")

    def text_to_vector(text, tfidf_dict):
        words = text.split()
        vector = np.zeros(len(tfidf_dict))
        for i, word in enumerate(tfidf_dict):
            if word in words:
                vector[i] = tfidf_dict[word]
        return vector

    # Menghitung representasi TF-IDF untuk seluruh data
    tfidf_dict = calculate_tfidf(calculate_tf(Data_ulasan["ulasan"]), calculate_idf(Data_ulasan["ulasan"]))

    # Mengonversi data ulasan pelatihan dan pengujian ke dalam vektor menggunakan representasi TF-IDF yang sama
    X_train_vectors = [text_to_vector(document, tfidf_dict) for document in X_train]
    X_test_vectors = [text_to_vector(document, tfidf_dict) for document in X_test]

    # Menampilkan Term Frequency (TF)
    st.write("Term Frequency (TF):")
#     for i, document in enumerate(X_train):
#         tf_dict = calculate_tf([document])
#         st.write(f"Document {i+1}:")
#         for word, tf in tf_dict.items():
#             st.write(f"{word}: {tf}")
#         st.write()

    # Menampilkan Document Frequency (DF)
    st.write("Document Frequency (DF):")
#     df_train = calculate_df(X_train)
#     for word, df in df_train.items():
#         st.write(f"{word}: {df}")
#     st.write()

    # Menampilkan Inverse Document Frequency (IDF)
    st.write("Inverse Document Frequency (IDF):")
#     for word, idf in idf_train.items():
#         st.write(f"{word}: {idf}")
#     st.write()

    # Klasifikasi menggunakan KNN
    k = 3
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train_vectors, y_train)

# Implementasi
with implementation:
# Implementasi dengan Streamlit
    st.title("Klasifikasi Sentimen Ulasan Menggunakan KNN")
    st.write("Masukkan ulasan di bawah ini:")
    input_text = st.text_input("Silahkan Masukkan Ulasan Anda :")

    if st.button("Prediksi"):
        # Mengubah input ulasan menjadi vektor
        input_vector = text_to_vector(input_text, tfidf_dict)
        input_vector = np.array(input_vector).reshape(1, -1)

        # Melakukan prediksi pada input ulasan
        predicted_label = knn_classifier.predict(input_vector)

        # Menampilkan hasil prediksi
        st.write("Hasil Prediksi:")
        st.write(f"Ulasan: {input_text}")
        st.write(f"Label: {predicted_label[0]}")

    # Menghitung akurasi pada data uji
    y_pred = knn_classifier.predict(X_test_vectors)
    accuracy = accuracy_score(y_test, y_pred)

    # Menampilkan akurasi
    st.write("Akurasi: {:.2f}%".format(accuracy * 100))

    # Menampilkan label prediksi
    st.write("Label Prediksi:")
    for i, (label, ulasan) in enumerate(zip(y_pred, X_test)):
        st.write(f"Data Uji {i+1}:")
        st.write(f"Ulasan: {ulasan}")
        st.write(f"Label: {label}")
        st.write()
