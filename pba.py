import streamlit as st
import pandas as pd
import numpy as np
import string
import re
from nltk.tokenize import RegexpTokenizer
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

st.title("PEMROSESAN BAHASA ALAMI A")
st.write("### Dosen Pengampu: Dr. FIKA HASTARITA RACHMAN, ST., M.Eng")
st.write("#### Kelompok: 5")
st.write("##### Hambali Fitrianto - 200411100074")
st.write("##### Pramudya Dwi Febrianto - 200411100042")
st.write("##### Febrian Achmad Syahputra - 200411100106")

# Load dataset
dataset = pd.read_csv("https://raw.githubusercontent.com/Feb11F/dataset/main/dieng_sentiment_pn.csv")

dataset['ulasan'] = dataset['ulasan'].str.lower()

def hapus_tweet_khusus(text):
    text = text.replace('\\t', " ").replace('\\n', " ").replace('\\u', " ").replace('\\', "")
    text = text.encode('ascii', 'replace').decode('ascii')
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)", " ", text).split())
    return text.replace("http://", " ").replace("https://", " ")

def hapus_nomor(text):
    return re.sub(r"\d+", "", text)

def hapus_tanda_baca(text):
    return text.translate(str.maketrans("", "", string.punctuation))

def hapus_whitespace_LT(text):
    return text.strip()

def hapus_whitespace_multiple(text):
    return re.sub('\s+', ' ', text)

def hapus_single_char(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)

def word_tokenize_wrapper(text):
    tokenizer = RegexpTokenizer(r'dataran\s+tinggi|jawa\s+tengah|[\w\']+')
    tokens = tokenizer.tokenize(text)
    return tokens

def freqDist_wrapper(text):
    return FreqDist(text)

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

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemmed_wrapper(term):
    return stemmer.stem(term)

term_dict = {}

for document in dataset['ulasan_tokens_WSW']:
    for term in document:
        if term not in term_dict:
            term_dict[term] = ' '

for term in term_dict:
    term_dict[term] = stemmed_wrapper(term)
    st.write(term, ":", term_dict[term])

def get_stemmed_term(document):
    return [term_dict[term] for term in document]

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


# Ambil inputan ulasan dari pengguna
input_ulasan = st.text_input("Masukkan ulasan:")

# Preprocessing ulasan
input_ulasan = input_ulasan.lower()
input_ulasan = hapus_tweet_khusus(input_ulasan)
input_ulasan = hapus_nomor(input_ulasan)
input_ulasan = hapus_tanda_baca(input_ulasan)
input_ulasan = hapus_whitespace_LT(input_ulasan)
input_ulasan = hapus_whitespace_multiple(input_ulasan)
input_ulasan = hapus_single_char(input_ulasan)
input_ulasan_tokens = word_tokenize_wrapper(input_ulasan)
input_ulasan_tokens_WSW = stopwords_removal(input_ulasan_tokens)
input_ulasan_tokens_stemmed = get_stemmed_term(input_ulasan_tokens)

# Load model yang telah dilatih sebelumnya
with open('model.pickle', 'rb') as file:
    loaded_model = pickle.load(file)

# Lakukan prediksi sentimen menggunakan model
input_ulasan_vectorized = tfidf_vectorizer.transform([' '.join(input_ulasan_tokens_stemmed)])
prediction = loaded_model.predict(input_ulasan_vectorized)

# Tampilkan hasil prediksi
if prediction == 0:
    st.write("Sentimen: Negatif")
else:
    st.write("Sentimen: Positif")

