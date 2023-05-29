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
from sklearn.naive_bayes import MultinomialNB
import pickle5 as pickle 
from sklearn.metrics import confusion_matrix, accuracy_score 

st.title("""
Aplikasi Analisis Sentimen Pendapat orang tua terhadap pembelajaran daring pada masa Covid-19 dengan algoritma naive bayes
""")

#Fractional Knapsack Problem
#Getting input from user
word = st.text_area('Masukkan kata yang akan di analisa :')

submit = st.button("submit")

if submit:
    def prep_input_data(word, slang_dict):
        lower_case_isi = word.lower()
        clean_symbols = re.sub("[^a-zA-Zï ]+"," ", lower_case_isi)
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
        return lower_case_isi,clean_symbols,slang,stem
    
    #Kamus
    with open('combined_slang_words.txt') as f:
        data = f.read()
    slang_dict = json.loads(data)

    #Dataset
    names = []
    with open(r'C:\Users\HP\test.txt', 'r') as fp:
        for line in fp:
            x = line[:-1]
            names.append(x)

    # TfidfVectorizer 
    tfidfvectorizer = TfidfVectorizer(analyzer='word')
    tfidf_wm = tfidfvectorizer.fit_transform(names)
    tfidf_tokens = tfidfvectorizer.get_feature_names_out()
    df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(),columns = tfidf_tokens)

    #Train test split
    training, test = train_test_split(tfidf_wm,test_size=0.2, random_state=1)#Nilai X training dan Nilai X testing
    training_label, test_label = train_test_split(df['Label'], test_size=0.2, random_state=1)#Nilai Y training dan Nilai Y testing    

    #model
    clf2 = MultinomialNB(alpha = 0.01)
    clf = clf2.fit(training, training_label)
    y_pred = clf.predict(test)

    #Evaluasi
    cm = confusion_matrix(test_label, y_pred) 
    akurasi = accuracy_score(test_label, y_pred)

    # #Inputan 
    lower_case_isi,clean_symbols,slang,stem = prep_input_data(word, slang_dict)
    
    #Prediksi
    v_data = tfidfvectorizer.transform([stem]).toarray()
    y_preds = clf2.predict(v_data)

    st.subheader('Preprocessing')
    st.write("Case Folding:",lower_case_isi)
    st.write("Cleansing :",clean_symbols)
    st.write("Slang Word :",slang)
    st.write("Steaming :",stem)

    st.subheader('Confussion Matrix')
    st.write(cm)

    st.subheader('Akurasi')
    st.info(akurasi)

    st.subheader('Prediksi')
    if y_preds == "Positif":
        st.success('Positive')
    else:
        st.error('Negative')
