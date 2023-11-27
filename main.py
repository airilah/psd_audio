import pickle
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
st.markdown(
    "<h1 style='text-align: center;'>Klasifikasi Kanker Payudara Menggunakan Model Random Forest</h1>", unsafe_allow_html=True
)
st.markdown(
    "<h4 style='text-align: center;'>Anas Khoiri Abdillah | 210411100025 | PSD - B</h4>", unsafe_allow_html=True
)


# load dataset -------------------------------------------------------------------
dataset_baru = pd.read_csv('dataset_baru.csv')

# memisahkan kolom fitur dan target
fitur = dataset_baru.drop(columns=['Classification'], axis=1)
target = dataset_baru['Classification']

# melakukan pembagian dataset, dataset dibagi menjadi 80% data training dan 20% data testing
fitur_train, fitur_test, target_train, target_test = train_test_split(fitur, target, test_size = 0.2, random_state=42)

# normalisasi dataset ------------------------------------------------------------

with open('zscorescaler_baru.pkl', 'rb') as file_normalisasi:
    zscore_scaler = pickle.load(file_normalisasi)
    
# menerapkan normalisasi zscore pada data training
zscore_training = zscore_scaler.transform(fitur_train)

# menerapkan normalisasi zscore pada data testing
zscore_testing = zscore_scaler.transform(fitur_test)

# implementasi data pda model

with open('gridrandomforestzscore.pkl', 'rb') as file_model:
    model_rf = pickle.load(file_model)
    
model_rf.fit(zscore_training, target_train)

st.warning("Masukkan hasil test Anda!")
Age = st.number_input ('Input Umur Anda! ')
Glucose = st.number_input ('Input hasil test Glukosa Anda!')
Homa = st.number_input ('Input hasil test Homa Anda!')
Adiponectin = st.number_input ('Input hasil test Adiponectin Anda!')
Resistin = st.number_input ('Input hasil test Resistin Anda!')


if st.button('Cek Status'):
    if Age is not None and Glucose is not None and Homa is not None and Adiponectin is not None and Resistin is not None:
        
        Age = int(Age)
        Glucose = float(Glucose)
        Homa = float(Homa)
        Adiponectin = float(Adiponectin)
        Resistin = float(Resistin)
        
        # Prediksi berdasarkan input yang telah diubah menjadi numerik
        prediksi = model_rf.predict([[Age, Glucose, Homa, Adiponectin, Resistin]])
        st.text( prediksi[0])
        if prediksi[0] == 1:
            st.success("Anda diprediksi Kanker Jinak / belum terdiagnosis Kanker Payudara!")
        else:
            st.error("Anda diprediksi Kanker Ganas / terdiagnosis Kanker Payudara")
    else:
        st.text('Data tidak boleh kosong. Harap isi semua kolom.')

