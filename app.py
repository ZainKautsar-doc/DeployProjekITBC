import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Load the saved model, scaler, and encoders
model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')
encoders = joblib.load('encoders.pkl')  # Memuat semua encoder dari satu file

# Title of the app
st.title("Prediksi Penyakit Jantung")

# Input fields
st.header("Masukkan Detail Pasien")
age = st.number_input("Usia", min_value=0, max_value=120, value=40)
sex = st.selectbox("Jenis Kelamin", ['M', 'F'])
chest_pain_type = st.selectbox("Tipe Nyeri Dada", ['ATA', 'NAP', 'ASY', 'TA'])
resting_bp = st.number_input("Tekanan Darah Istirahat (mmHg)", min_value=0, max_value=300, value=140)
cholesterol = st.number_input("Kolesterol (mg/dl)", min_value=0, max_value=1000, value=200)
fasting_bs = st.selectbox("Gula Darah Puasa (> 120 mg/dl)", [0, 1])
resting_ecg = st.selectbox("ECG Istirahat", ['Normal', 'ST', 'LVH'])
max_hr = st.number_input("Detak Jantung Maksimum", min_value=0, max_value=300, value=150)
exercise_angina = st.selectbox("Angina saat Olahraga", ['N', 'Y'])
oldpeak = st.number_input("Oldpeak (Depresi ST)", min_value=-2.6, max_value=6.2, value=0.0, step=0.1)
st_slope = st.selectbox("Kemiringan ST", ['Up', 'Flat', 'Down'])

# Prepare input data
input_data = pd.DataFrame({
    'Age': [age],
    'Sex': [sex],
    'ChestPainType': [chest_pain_type],
    'RestingBP': [resting_bp],
    'Cholesterol': [cholesterol],
    'FastingBS': [fasting_bs],
    'RestingECG': [resting_ecg],
    'MaxHR': [max_hr],
    'ExerciseAngina': [exercise_angina],
    'Oldpeak': [oldpeak],
    'ST_Slope': [st_slope]
})

# Encode categorical variables using the loaded encoders
input_data['Sex'] = encoders['Sex'].transform(input_data['Sex'])
input_data['ChestPainType'] = encoders['ChestPainType'].transform(input_data['ChestPainType'])
input_data['RestingECG'] = encoders['RestingECG'].transform(input_data['RestingECG'])
input_data['ExerciseAngina'] = encoders['ExerciseAngina'].transform(input_data['ExerciseAngina'])
input_data['ST_Slope'] = encoders['ST_Slope'].transform(input_data['ST_Slope'])

# Scale numerical features
numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

# Predict button
if st.button("Prediksi"):
    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)[0]  # Mendapatkan probabilitas untuk kedua kelas

    # Probabilitas untuk kelas 0 (Tidak memiliki penyakit jantung) dan kelas 1 (Memiliki penyakit jantung)
    prob_no_disease = probabilities[0] * 100
    prob_disease = probabilities[1] * 100

    st.subheader("Hasil Prediksi")
    if prob_disease < 30:
        st.success("Pasien diprediksi TIDAK memiliki Penyakit Jantung.")
    elif 30 <= prob_disease <= 50:
        st.warning("Pasien memiliki Resiko Sedang terhadap Penyakit Jantung.")
    else:  # prob_disease > 50
        st.error("Pasien diprediksi Beresiko memiliki Penyakit Jantung.")

    # Tampilkan probabilitas untuk kedua kelas
    st.write(f"*Probabilitas Tidak Memiliki Penyakit Jantung:* {prob_no_disease:.2f}%")
    st.write(f"*Probabilitas Memiliki Penyakit Jantung:* {prob_disease:.2f}%")

# Add a note
st.markdown("*Catatan:* Ini adalah alat prediksi berbasis model Random Forest yang dilatih dengan SMOTE dan GridSearchCV. Konsultasikan dengan tenaga medis profesional untuk saran medis.")