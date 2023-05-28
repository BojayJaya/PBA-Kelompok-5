import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier

st.title("PEMROSESAN BAHASA ALAMI A")
st.write("### Dosen Pengampu : Dr. FIKA HASTARITA RACHMAN, ST., M.Eng")
st.write("#### Kelompok : 5")
st.write("##### Hambali Fitrianto - 200411100074")
st.write("##### Pramudya Dwi Febrianto - 200411100042")
st.write("##### Febrian Achmad Syahputra - 200411100106")

#Navbar
data_set_description, upload_data, preprocessing, modeling, implementation = st.tabs(["Data Set Description", "Upload Data", "Preprocessing", "Modeling", "Implementation"])

df = pd.read_csv('https://raw.githubusercontent.com/HambaliFitrianto/Aplikasi-Web-Data-Mining-Weather-Prediction/main/seattle-weather.csv')

#data_set_description
with data_set_description:
    st.write("###### Judul : ANALISIS SENTIMEN PADA WISATA DIENG DENGAN ALGORITMA K-NEAREST NEIGHBOR (K-NN) ")
    st.write("""###### Penjelasan Prepocessing Data : """)
    st.write("""1. Case Folding :
    
    Case folding adalah proses dalam pemrosesan teks yang mengubah semua huruf dalam teks menjadi huruf kecil atau huruf besar. Tujuan dari case folding adalah untuk mengurangi variasi yang disebabkan oleh perbedaan huruf besar dan kecil dalam teks, sehingga mempermudah pemrosesan teks secara konsisten.
    
    Dalam case folding, biasanya semua huruf dalam teks dikonversi menjadi huruf kecil dengan menggunakan metode seperti lowercasing. Dengan demikian, perbedaan antara huruf besar dan huruf kecil tidak lagi diperhatikan dalam analisis teks, sehingga memungkinkan untuk mendapatkan hasil yang lebih konsisten dan mengurangi kompleksitas dalam pemrosesan teks.
    """)
    st.write("""2. Tokenize :

    Tokenisasi adalah proses pemisahan teks menjadi unit-unit yang lebih kecil yang disebut token. Token dapat berupa kata, frasa, atau simbol lainnya, tergantung pada tujuan dan aturan tokenisasi yang digunakan.

    Tujuan utama tokenisasi dalam pemrosesan bahasa alami (Natural Language Processing/NLP) adalah untuk memecah teks menjadi unit-unit yang lebih kecil agar dapat diolah lebih lanjut, misalnya dalam analisis teks, pembentukan model bahasa, atau klasifikasi teks.
    """)
    st.write("""3. Filtering (Stopword Removal) :

    Filtering atau Stopword Removal adalah proses penghapusan kata-kata yang dianggap tidak memiliki makna atau kontribusi yang signifikan dalam analisis teks. Kata-kata tersebut disebut sebagai stop words atau stopwords.

    Stopwords biasanya terdiri dari kata-kata umum seperti “a”, “an”, “the”, “is”, “in”, “on”, “and”, “or”, dll. Kata-kata ini sering muncul dalam teks namun memiliki sedikit kontribusi dalam pemahaman konten atau pengambilan informasi penting dari teks.

    Tujuan dari Filtering atau Stopword Removal adalah untuk membersihkan teks dari kata-kata yang tidak penting sehingga fokus dapat diarahkan pada kata-kata kunci yang lebih informatif dalam analisis teks. Dengan menghapus stopwords, kita dapat mengurangi dimensi data, meningkatkan efisiensi pemrosesan, dan memperbaiki kualitas hasil analisis.
    """)
    st.write("""4. Stemming :

    Stemming dalam pemrosesan bahasa alami (Natural Language Processing/NLP) adalah proses mengubah kata ke dalam bentuk dasarnya atau bentuk kata yang lebih sederhana, yang disebut sebagai “stem”. Stemming bertujuan untuk menghapus infleksi atau imbuhan pada kata sehingga kata-kata yang memiliki akar kata yang sama dapat diidentifikasi sebagai bentuk yang setara.
    """)
    
    
    st.write("""5. weather (cuaca) :

    Output (keluaran)
    """)
    st.write("""Menggunakan Kolom (input) :

    precipitation
    tempmax * tempmin
    wind
    """)
    st.write("""Memprediksi kondisi cuaca (output) :

    1. drizzle (gerimis)
    2. rain (hujan)
    3. sun (matahari)
    4. snow (salju)
    5. fog (kabut)
    """)
    st.write("###### Aplikasi ini untuk : Weather Prediction (Prediksi Cuaca) ")
    st.write("###### Source Code Aplikasi ada di Github anda bisa acces di link : https://github.com/HambaliFitrianto/Aplikasi-Web-Data-Mining-Weather-Prediction")
    st.write("###### Untuk Wa saya anda bisa hubungi nomer ini : http://wa.me/6282138614807 ")

#Uploud data
with upload_data:
    uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        st.write("Nama File Anda = ", uploaded_file.name)
        # view dataset asli
        st.header("Dataset")
        st.dataframe(df)

#Preprocessing
with preprocessing:
    st.subheader("""Normalisasi Data""")
    st.write("""Rumus Normalisasi Data :""")
    st.image('https://i.stack.imgur.com/EuitP.png', use_column_width=False, width=250)
    st.markdown("""
    Dimana :
    - X = data yang akan dinormalisasi atau data asli
    - min = nilai minimum semua data asli
    - max = nilai maksimum semua data asli
    """)
    df = df.drop(columns=["date"])
    #Mendefinisikan Varible X dan Y
    X = df.drop(columns=['weather'])
    y = df['weather'].values
    df_min = X.min()
    df_max = X.max()
    
    #NORMALISASI NILAI X
    scaler = MinMaxScaler()
    #scaler.fit(features)
    #scaler.transform(features)
    scaled = scaler.fit_transform(X)
    features_names = X.columns.copy()
    #features_names.remove('label')
    scaled_features = pd.DataFrame(scaled, columns=features_names)

    st.subheader('Hasil Normalisasi Data')
    st.write(scaled_features)

    st.subheader('Target Label')
    dumies = pd.get_dummies(df.weather).columns.values.tolist()
    dumies = np.array(dumies)

    labels = pd.DataFrame({
        '1' : [dumies[0]],
        '2' : [dumies[1]],
        '3' : [dumies[2]],
        '4' : [dumies[3]],
        '5' : [dumies[4]]
    })

    st.write(labels)

#Modelling
with modeling:
    training, test = train_test_split(scaled_features,test_size=0.2, random_state=1)#Nilai X training dan Nilai X testing
    training_label, test_label = train_test_split(y, test_size=0.2, random_state=1)#Nilai Y training dan Nilai Y testing
    with st.form("modeling"):
        st.subheader('Modeling')
        st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
        # naive = st.checkbox('Gaussian Naive Bayes')
        k_nn = st.checkbox('K-Nearest Neighboor')
        # destree = st.checkbox('Decission Tree')
        submitted = st.form_submit_button("Submit")

        # NB
        # GaussianNB(priors=None)

        # Fitting Naive Bayes Classification to the Training set with linear kernel
        # gaussian = GaussianNB()
        # gaussian = gaussian.fit(training, training_label)

        # Predicting the Test set results
        # y_pred = gaussian.predict(test)
    
        # y_compare = np.vstack((test_label,y_pred)).T
        # gaussian.predict_proba(test)
        # gaussian_akurasi = round(100 * accuracy_score(test_label, y_pred))
        # akurasi = 10

        #KNN
        K=5
        knn=KNeighborsClassifier(n_neighbors=K)
        knn.fit(training,training_label)
        knn_predict=knn.predict(test)

        knn_akurasi = round(100 * accuracy_score(test_label,knn_predict))

        #Decission Tree
        # dt = DecisionTreeClassifier()
        # dt.fit(training, training_label)
        # prediction
        # dt_pred = dt.predict(test)
        #Accuracy
        # dt_akurasi = round(100 * accuracy_score(test_label,dt_pred))

        if submitted :
            # if naive :
            #     st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(gaussian_akurasi))
            if k_nn :
                st.write("Model KNN accuracy score : {0:0.2f}" . format(knn_akurasi))
            # if destree :
            #     st.write("Model Decision Tree accuracy score : {0:0.2f}" . format(dt_akurasi))
        
        grafik = st.form_submit_button("Grafik akurasi semua model")
        if grafik:
            data = pd.DataFrame({
                # 'Akurasi' : [gaussian_akurasi, knn_akurasi, dt_akurasi],
                # 'Model' : ['Gaussian Naive Bayes', 'K-NN', 'Decission Tree'],
                'Akurasi' : [knn_akurasi],
                'Model' : ['K-NN'],
            })

            chart = (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    alt.X("Akurasi"),
                    alt.Y("Model"),
                    alt.Color("Akurasi"),
                    alt.Tooltip(["Akurasi", "Model"]),
                )
                .interactive()
            )
            st.altair_chart(chart,use_container_width=True)

#Implementasi
with implementation:
    with st.form("my_form"):
        st.subheader("Implementasi")
        Ulasan = st.number_input('Masukkan preciptation (curah hujan) : ')
        model = st.selectbox('Pilihlah model yang akan anda gunakan untuk melakukan prediksi?',
                ('K-NN'))

        prediksi = st.form_submit_button("Submit")
        if prediksi:
            inputs = np.array([
                Ulasan,
            ])

            df_min = X.min()
            df_max = X.max()
            input_norm = ((inputs - df_min) / (df_max - df_min))
            input_norm = np.array(input_norm).reshape(1, -1)

            if model == 'K-NN':
                mod = knn 

            input_pred = mod.predict(input_norm)

            st.subheader('Hasil Prediksi')
            st.write('Menggunakan Pemodelan :', model)

            st.write(input_pred)
# with implementation:
#     with st.form("my_form"):
#         st.subheader("Implementasi")
#         Precipitation = st.number_input('Masukkan preciptation (curah hujan) : ')
#         Temp_Max = st.number_input('Masukkan tempmax (suhu maks) : ')
#         Temp_Min = st.number_input('Masukkan tempmin (suhu min) : ')
#         Wind = st.number_input('Masukkan wind (angin) : ')
#         model = st.selectbox('Pilihlah model yang akan anda gunakan untuk melakukan prediksi?',
#                 ('Gaussian Naive Bayes', 'K-NN', 'Decision Tree'))

#         prediksi = st.form_submit_button("Submit")
#         if prediksi:
#             inputs = np.array([
#                 Precipitation,
#                 Temp_Max,
#                 Temp_Min,
#                 Wind
#             ])

#             df_min = X.min()
#             df_max = X.max()
#             input_norm = ((inputs - df_min) / (df_max - df_min))
#             input_norm = np.array(input_norm).reshape(1, -1)

#             if model == 'Gaussian Naive Bayes':
#                 mod = gaussian
#             if model == 'K-NN':
#                 mod = knn 
#             if model == 'Decision Tree':
#                 mod = dt

#             input_pred = mod.predict(input_norm)

#             st.subheader('Hasil Prediksi')
#             st.write('Menggunakan Pemodelan :', model)

#             st.write(input_pred)
