import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
import keras   
from tensorflow.keras.models import load_model


st.set_page_config(layout="wide")

#jika terdapat eror aktifkan ini 
original_dense_init = keras.layers.Dense.__init__
#fungsi modifikasi untuk membuang quantization_config jika tidak dikenali
def patched_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)  # Hapus paksa parameter pemicu eror
    original_dense_init(self, *args, **kwargs)
    
# Ganti fungsi asli Keras dengan fungsi modifikasi
keras.layers.Dense.__init__ = patched_dense_init



@st.cache_resource
def load_models():
    ml_model = joblib.load("models/random_forest_regressor.joblib")
    ann_model = load_model("models/ann_model.keras", compile=False)
    ml_scaler = joblib.load("models/ml_scaler.joblib")
    ann_scaler = joblib.load("models/ann_scaler.joblib")
    return ml_model, ann_model, ml_scaler, ann_scaler

ml_model, ann_model, ml_scaler, ann_scaler = load_models()


st.title("Calories Burned 🔥")
st.markdown('----')

st.subheader('Sudah olahraga?')
st.write('Saanya hitung kalori lo yg terbakar selama lo olahraga')
st.markdown('----')


col1, col2 = st.columns(2)

# INPUTAN
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    duration = st.number_input("Duration (minutes)", 1, 300, 30)
    weight = st.number_input("Weight (kg)", 30, 200, 60) 

with col2:
    age = st.slider("Age", 10, 100, 25)
    heart_rate = st.slider("Heart Rate", 40, 220, 100)
    height = st.number_input("Height (cm)", 100, 250, 165) 

model_choice = st.selectbox(
    "Pilih Model",
    ["Random Forest", "ANN"]
)

# PROSES MODEL
gender_encoded = 1 if gender == "Male" else 0

input_data = pd.DataFrame({
    "Gender": [gender_encoded],
    "Age": [age],
    "Duration": [duration],
    "Heart_Rate": [heart_rate]
})


# Hitung BMI
if st.button("Hitung"):

    height_meters = height / 100
    bmi_score = weight / (height_meters ** 2)
    
    if bmi_score < 18.5:
        bmi_status, bmi_color = "Underweight (Kurus)", "orange"
    elif 18.5 <= bmi_score < 25:
        bmi_status, bmi_color = "Normal (Ideal)", "green"
    elif 25 <= bmi_score < 30:
        bmi_status, bmi_color = "Overweight (Gemuk)", "orange"
    else:
        bmi_status, bmi_color = "Obesity (Obesitas)", "red"
        
    st.markdown("### 📊 Hasil Analisis")
    

    #Output
    out_col1, out_col2 = st.columns(2)
    with out_col1:
        st.markdown(f"##### 🔥 Prediksi Kalori ({model_choice})")
        
        if model_choice == "Random Forest":
            input_scaled = ml_scaler.transform(input_data)
            pred = ml_model.predict(input_scaled)[0] 
            st.success(f"**{pred:.2f} kcal** terbakar")
        else:
            input_scaled = ann_scaler.transform(input_data)
            pred = ann_model.predict(input_scaled, verbose=0)[0][0]
            st.success(f"**{pred:.2f} kcal** terbakar")

    with out_col2:
        st.markdown("##### 🩺 Status Komposisi Tubuh")
        st.info(f"**BMI:** {bmi_score:.1f}Status: :{bmi_color}[**{bmi_status}**]")