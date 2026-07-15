import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model.pkl")
imputer = joblib.load("imputer.pkl")
encoders = joblib.load("encoders.pkl")

st.set_page_config(
    page_title="Prediksi Tingkat Stres Mahasiswa",
    page_icon="🧠",
    layout="centered"
)

# Sidebar
st.sidebar.title("Tentang Aplikasi")

st.sidebar.info("""
### Prediksi Tingkat Stres Mahasiswa

**Algoritma :**
Random Forest Classifier

**Hyperparameter Tuning :**
RandomizedSearchCV
""")

# Judul
st.title("🧠 Prediksi Tingkat Stres Mahasiswa")

st.write(
    """
Masukkan data mahasiswa pada form di bawah ini.
Sistem akan memprediksi apakah mahasiswa mengalami **Stres** atau **Tidak Stres**.
"""
)

st.divider()

# Input
pilihan_mahasiswa = {
    "Mahasiswa Perguruan Tinggi": "college",
    "Siswa Sekolah": "school",
    "Mahasiswa Sambil Bekerja": "working_student"
}

jenis_mahasiswa = st.selectbox(
    "Jenis Mahasiswa",
    list(pilihan_mahasiswa.keys())
)

jam_tidur = st.slider(
    "Jam Tidur",
    min_value=2.0,
    max_value=10.0,
    value=7.0,
    step=0.5
)

jam_belajar = st.slider(
    "Jam Belajar",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)

jam_media_sosial = st.slider(
    "Jam Penggunaan Media Sosial",
    min_value=0.0,
    max_value=10.0,
    value=3.0,
    step=0.5
)

kehadiran = st.slider(
    "Persentase Kehadiran",
    min_value=0,
    max_value=100,
    value=80
)

tekanan_ujian = st.slider(
    "Tingkat Tekanan Ujian",
    min_value=1,
    max_value=10,
    value=5
)

dukungan_keluarga = st.slider(
    "Tingkat Dukungan Keluarga",
    min_value=1,
    max_value=10,
    value=6
)

bulan = st.selectbox(
    "Bulan",
    options=[1,2,3,4,5,6,7,8,9,10,11,12],
    format_func=lambda x: [
        "Januari",
        "Februari",
        "Maret",
        "April",
        "Mei",
        "Juni",
        "Juli",
        "Agustus",
        "September",
        "Oktober",
        "November",
        "Desember"
    ][x-1]
)

st.divider()

# Prediksi
if st.button("Prediksi Tingkat Stres"):

    jenis_mahasiswa = pilihan_mahasiswa[jenis_mahasiswa]

    jenis_mahasiswa = encoders["Student_Type"].transform(
        [jenis_mahasiswa]
    )[0]

    data = pd.DataFrame({
        "Student_Type":[jenis_mahasiswa],
        "Sleep_Hours":[jam_tidur],
        "Study_Hours":[jam_belajar],
        "Social_Media_Hours":[jam_media_sosial],
        "Attendance":[kehadiran],
        "Exam_Pressure":[tekanan_ujian],
        "Family_Support":[dukungan_keluarga],
        "Month":[bulan]
    })

    hasil = model.predict(data)[0]
    probabilitas = model.predict_proba(data)[0]

    st.subheader("📋 Hasil Prediksi")

    if hasil == 1:
        st.error("⚠️ Mahasiswa diprediksi mengalami **Stres**")
    else:
        st.success("✅ Mahasiswa diprediksi **Tidak Mengalami Stres**")

    st.divider()

    st.subheader("📊 Probabilitas Prediksi")

    st.write(f"**Tidak Stres : {probabilitas[0]*100:.2f}%**")
    st.progress(float(probabilitas[0]))

    st.write(f"**Stres : {probabilitas[1]*100:.2f}%**")
    st.progress(float(probabilitas[1]))