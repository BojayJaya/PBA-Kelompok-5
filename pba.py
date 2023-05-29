import streamlit as st
import pandas as pd
import numpy as np
import regex as re
import json
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score
import requests

st.title("Aplikasi Analisis Sentimen Pendapat orang tua terhadap pembelajaran daring pada masa Covid-19 dengan algoritma KNN")

# Fractional Knapsack Problem
# Getting input from user
word = st.text_area("Masukkan kata yang akan dianalisis:")

submit = st.button("Submit")

if submit:
    def prep_input_data(word, slang_dict):
        lower_case_isi = word.lower()
        clean_symbols = re.sub("[^a-zA-Zï ]+", " ", lower_case_isi)
        
        def replace_slang_words(text):
            words = nltk.word_tokenize(text.lower())
            words_filtered = [word for word in words if word not in stopwords.words('indonesian')]
            for i in range(len(words_filtered)):
                if words_filtered[i] in slang_dict:
                    words_filtered[i] = slang_dict[words_filtered[i]]
            return ' '.join(words_filtered)
        
        slang = replace_slang_words(clean_symbols)
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        stem = stemmer.stem(slang)
        return lower_case_isi, clean_symbols, slang, stem

    # Kamus
    with open('combined_slang_words.txt') as f:
        data = f.read()
    slang_dict = json.loads(data)

    # Load data.pickle from GitHub
    url = 'https://raw.githubusercontent.com/your_username/your_repository/your_branch/data.pickle'
    response = requests.get(url)
    with open('data.pickle', 'wb') as f:
        f.write(response.content)

    # Load the pickled data
    with open('data.pickle', 'rb') as f:
        data = pickle.load(f)
    
    names = data['names']
    df = data['df']

    # TfidfVectorizer
    tfidfvectorizer = TfidfVectorizer(analyzer='word')
    tfidf_wm = tfidfvectorizer.fit_transform(names)
    tfidf_tokens = tfidfvectorizer.get_feature_names_out()
    df_tfidfvect = pd.DataFrame(data=tfidf_wm.toarray(), columns=tfidf_tokens)

    # Train test split
    training, test = train_test_split(tfidf_wm, test_size=0.2, random_state=1)
    training_label, test_label = train_test_split(df['Label'], test_size=0.2, random_state=1)

    # Model
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(training, training_label)
    y_pred = clf.predict(test)

    # Evaluasi
    cm = confusion_matrix(test_label, y_pred)
    akurasi = accuracy_score(test_label, y_pred)

    # Inputan
    lower_case_isi, clean_symbols, slang, stem = prep_input_data(word, slang_dict)
    
    # Prediksi
    v_data = tfidfvectorizer.transform([stem]).toarray()
    y_preds = clf.predict(v_data)
        st.subheader('Preprocessing')
    st.write("Case Folding:", lower_case_isi)
    st.write("Cleansing:", clean_symbols)
    st.write("Slang Word:", slang)
    st.write("Steaming:", stem)

    st.subheader('Confusion Matrix')
    st.write(cm)

    st.subheader('Akurasi')
    st.info(akurasi)

    st.subheader('Prediksi')
    if y_preds == "Positif":
        st.success('Positive')
    else:
        st.error('Negative')
