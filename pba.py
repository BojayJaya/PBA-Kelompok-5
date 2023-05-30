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


    # dataset = pd.read_csv('https://raw.githubusercontent.com/BojayJaya/PBA-Kelompok-5/main/Text_Preprocessing.csv')

    Data_ulasan = pd.read_csv("https://raw.githubusercontent.com/BojayJaya/PBA-Kelompok-5/main/hasil_preprocessing.csv")

    Data_ulasan = pd.DataFrame(Data_ulasan)
    Data_ulasan.head()

    ulasan = Data_ulasan['ulasan_hasil_preprocessing']
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

    Data_ulasan["ulasan_list"] = Data_ulasan["ulasan_hasil_preprocessing"].apply(convert_text_list)
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

    # Menampilkan Document Frequency (DF)
    st.write("Document Frequency (DF):")

    # Menampilkan Inverse Document Frequency (IDF)
    st.write("Inverse Document Frequency (IDF):")

    # Klasifikasi menggunakan KNN
    k = 3
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train_vectors, y_train)


    input_vector = text_to_vector(ulasan, tfidf_dict)
    input_vector = np.array(input_vector).reshape(1, -1)

    # Melakukan prediksi pada input ulasan
    predicted_label = knn_classifier.predict(input_vector)

    # Menampilkan hasil prediksi
    st.write("Hasil Prediksi:")
    st.write(f"Ulasan: {ulasan}")
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
