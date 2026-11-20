import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.features import clean, RAWCOLS

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "model.joblib"
META_PATH = BASE_DIR / "artifacts" / "metadata.json"

st.set_page_config(
    page_title="Injection Molding Quality Control",
    page_icon="🏭",
    layout="wide",
)

@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, metadata

st.title("🏭 Injection Molding Quality Control")
st.caption("Sistem prediksi risiko cacat komponen hasil proses injection molding")
st.write(
    "Masukkan parameter dari satu siklus produksi. Sistem akan menghitung "
    "probabilitas cacat berdasarkan model machine learning."
)

if not MODEL_PATH.exists() or not META_PATH.exists():
    st.error(
        "File model atau metadata belum ditemukan. Pastikan model.joblib dan "
        "metadata.json berada di folder artifacts/."
    )
    st.stop()

try:
    model, metadata = load_assets()
except Exception as exc:
    st.error(f"Gagal memuat model atau metadata: {exc}")
    st.stop()

threshold = float(metadata["threshold"])

with st.form("prediction_form"):
    st.subheader("Parameter produksi")

    c1, c2, c3 = st.columns(3)
    with c1:
        mesin = st.selectbox("Mesin", ["IM-01", "IM-02", "IM-03", "IM-04"])
        shift = st.selectbox("Shift", ["pagi", "siang", "malam"])
        bahan = st.selectbox("Bahan", ["PP", "ABS", "PC"])
        suhu_leleh = st.number_input("Suhu leleh (°C)", min_value=0.0, max_value=700.0, value=245.0, step=1.0)
    with c2:
        suhu_cetakan = st.number_input("Suhu cetakan (°C)", min_value=0.0, max_value=250.0, value=62.0, step=1.0)
        tekanan = st.number_input("Tekanan injeksi (bar)", min_value=0.0, max_value=3000.0, value=760.0, step=10.0)
        kecepatan = st.number_input("Kecepatan injeksi (mm/s)", min_value=0.0, max_value=1000.0, value=80.0, step=1.0)
    with c3:
        pendinginan = st.number_input("Waktu pendinginan (detik)", min_value=0.0, max_value=300.0, value=11.0, step=0.5)
        siklus = st.number_input("Waktu siklus (detik)", min_value=0.0, max_value=1000.0, value=26.0, step=0.5)
        kelembapan = st.number_input("Kelembapan bahan (%)", min_value=0.0, max_value=100.0, value=0.18, step=0.01, format="%.2f")

    submitted = st.form_submit_button("Prediksi kualitas produk", type="primary", use_container_width=True)

if submitted:
    input_data = pd.DataFrame([{
        "suhu_leleh_C": suhu_leleh,
        "suhu_cetakan_C": suhu_cetakan,
        "tekanan_injeksi_bar": tekanan,
        "kecepatan_injeksi_mm_s": kecepatan,
        "waktu_pendinginan_s": pendinginan,
        "waktu_siklus_s": siklus,
        "kelembaban_bahan_pct": kelembapan,
        "mesin": mesin,
        "shift": shift,
        "bahan": bahan,
    }])

    try:
        cleaned = clean(input_data, drop_duplicates=False)
        probability = float(model.predict_proba(cleaned[RAWCOLS])[:, 1][0])
        flagged = probability >= threshold

        st.divider()
        st.subheader("Hasil prediksi")
        left, right = st.columns(2)
        left.metric("Probabilitas cacat", f"{probability * 100:.2f}%")
        right.metric("Threshold keputusan", f"{threshold * 100:.2f}%")

        if flagged:
            st.error("Produk terindikasi berisiko cacat")
            st.write(
                "Saran: pisahkan atau tahan komponen untuk pemeriksaan QC. "
                "Periksa kembali parameter proses dan konsultasikan penyesuaian "
                "dengan tim proses sebelum mengubah pengaturan mesin."
            )
        else:
            st.success("Produk tidak ditandai oleh model sebagai berisiko cacat")
            st.write(
                "Tetap ikuti prosedur inspeksi mutu yang berlaku. "
                "Prediksi model bukan jaminan bahwa produk bebas cacat."
            )

        st.caption(
            "Hasil merupakan prediksi probabilistik, bukan keputusan QC final. "
            "Pastikan satuan dan nilai parameter sesuai kondisi mesin."
        )
    except Exception as exc:
        st.error(f"Prediksi gagal. Periksa kembali nilai masukan. Detail: {exc}")

with st.expander("Informasi model"):
    st.write(f"**Algoritma:** {metadata.get('algoritma', 'Tidak tercantum')}")
    st.write(f"**Versi model:** {metadata.get('versi_model', '-')}")
    st.write(f"**Threshold:** {threshold:.4f}")
    st.write(f"**Target:** {metadata.get('target', 'cacat')}")
