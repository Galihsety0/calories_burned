import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model


# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Calories Burned",
    page_icon="🔥",
    layout="wide"
)

# LOAD MODEL & SCALER
# =========================================================
@st.cache_resource
def load_ann_model():
    model = load_model('models/calories_model.h5', compile=False)
    scaler = joblib.load('models/scaler_ann.pkl')
    return model, scaler

@st.cache_resource
def load_rf_model():
    model = joblib.load('models/rf_model.pkl')
    scaler = joblib.load('models/scaler_ml.pkl')
    return model, scaler

try:
    ann_model, ann_scaler = load_ann_model()
    rf_model, rf_scaler = load_rf_model()
except Exception as e:
    st.error()
    st.exception(e)
    st.stop()

# FUNGSI BANTU
# =========================================================
def hitung_bmi(berat_kg: float, tinggi_cm: float) -> float:
    tinggi_m = tinggi_cm / 100
    return berat_kg / (tinggi_m ** 2)

def kategori_bmi(bmi: float) -> tuple[str, str]:
    
    if bmi < 18.5:
        return "Kurus (Underweight)", "blue"
    elif bmi < 23:
        return "Normal", "green"
    elif bmi < 25:
        return "Berat Badan Lebih (Overweight)", "orange"
    else:
        return "Obesitas", "red"

def preprocess_input(gender_val: int, age: float, duration: float, heart_rate: float, scaler) -> np.ndarray:
    raw = np.array([[gender_val, age, duration, heart_rate]])
    scaled = scaler.transform(raw)
    return scaled


# HEADER
# =========================================================
st.title("Calories Burned")
st.markdown(
    "Aplikasi ini dibuat dengan menggunakan dua pilihan model: **Random Forest Regressor** (Machine "
    "Learning) atau **Artificial Neural Network (ANN)**, berdasarkan data "
    "Gender, Usia, Durasi olahraga, dan Heart Rate.\n\n"
    "Selain itu, dapat juga menghitung **BMI (Body Mass Index)** lo masuk kategori apa, ini sebagai "
    "info tambahan saja— *BMI tidak digunakan dalam prediksi kalori model*."
)

st.divider()


st.html("<h1 style='text-align: center;'>Saatnya hitung kalori lo yg berhasil lo bakarrr saat berolahraga!! 🔥</h2>")



# MODEL
# =========================================================
st.subheader("Pilih Model")
model_choice = st.selectbox(
    "Model yang digunakan untuk hitung kalori lo",
    options=["Random Forest Regressor (ML)", "Artificial Neural Network (ANN)"]
)

with st.expander("ℹ️ Perbandingan performa model (hasil evaluasi)"):
    perf_df = pd.DataFrame({
        "Model": ["Random Forest Regressor", "Artificial Neural Network"],
        "R² Score (Test)": ["99.66%", "99.70%"],
        "MAE (Test)": ["2.54 kcal", "2.39 kcal"],
        "RMSE (Test)": ["3.70", "3.43"],
    })
    st.dataframe(perf_df, hide_index=True, use_container_width=True)


# FORM INPUT
# =========================================================
st.subheader("📋 Input Data Diri & Aktivitas Lo")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.pills("Gender", options=["Male", "Female"])
    age = st.number_input("Usia (tahun)", min_value=10, max_value=100, value=25, step=1)

with col2:
    duration = st.number_input("Durasi Olahraga (menit)", min_value=1.0, max_value=300.0, value=30.0, step=1.0)
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=40.0, max_value=220.0, value=120.0, step=1.0)

with col3: 
    tinggi = st.slider("Tinggi Badan (cm)", min_value=100.0, max_value=250.0, value=165.0, step=0.5)
    berat = st.slider("Berat Badan (kg)", min_value=20.0, max_value=250.0, value=60.0, step=0.5)

st.divider()


# PREDIKSI
# =========================================================
if st.button("🔍 Hitung ", use_container_width=True):

    # --- Encoding Gender (sesuai notebook: male=1, female=0) ---
    gender_val = 1 if gender == "Male" else 0

    # --- Pilih model & scaler sesuai dropdown ---
    if model_choice == "Random Forest Regressor (ML)":
        active_scaler = rf_scaler
        X_scaled = preprocess_input(gender_val, age, duration, heart_rate, active_scaler)
        kalori = float(rf_model.predict(X_scaled)[0])
    else:
        active_scaler = ann_scaler
        X_scaled = preprocess_input(gender_val, age, duration, heart_rate, active_scaler)
        prediksi = ann_model.predict(X_scaled, verbose=0)
        kalori = float(prediksi[0][0])

    # --- Hitung BMI (terpisah dari model) ---
    bmi = hitung_bmi(berat, tinggi)
    label_bmi, warna_bmi = kategori_bmi(bmi)


    # HASIL
    # =========================================================
    st.subheader("📊 Hasil")
    st.caption(f"Model yang digunakan: **{model_choice}**")

    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.metric(label="Estimasi Kalori Terbakar", value=f"{kalori:,.1f} kcal")

    with res_col2:
        st.metric(label="BMI Lo", value=f"{bmi:.1f}")
        st.markdown(f"Kategori: **:{warna_bmi}[{label_bmi}]**")

    st.caption(
        "Note: Hasil prediksi merupakan estimasi berdasarkan model machine learning dan ANN "
        "tidak dapat menggantikan pengukuran medis."
    )
