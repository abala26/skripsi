# 🍃 Aplikasi Klasifikasi Penyakit Daun Teh (Streamlit)

Aplikasi web interaktif berbasis Streamlit untuk mendeteksi dan mengklasifikasikan penyakit pada daun teh dari citra/gambar, berdasarkan hasil penelitian dan eksperimen pada `skripsi (2).ipynb`.

---

## 🌟 Fitur Utama

1. **Dukungan Dua Arsitektur Deep Learning**:
   - **ResNet-50** (Convolutional Neural Network) - Akurasi Uji: **93.28%**
   - **Vision Transformer (ViT-B/16)** (Self-Attention Transformer) - Akurasi Uji: **91.04%**
   - **Mode Komparasi Berdampingan (Side-by-Side Comparison)**: Membandingkan prediksi, tingkat keyakinan (*confidence*), dan latensi komputasi kedua model sekaligus.
2. **Fleksibilitas Input Citra**:
   - Unggah berkas gambar (`.jpg`, `.jpeg`, `.png`, `.webp`)
   - Kamera langsung (*Webcam capture*) via `st.camera_input`
3. **Analisis Prediksi Lengkap**:
   - Indikator diagnosis & tingkat keyakinan (*Confidence Score*)
   - Peringatan jika tingkat keyakinan di bawah ambang batas (*threshold slider*)
   - Diagram batang distribusi probabilitas untuk seluruh 8 kelas
   - Waktu inferensi (*latency*) dalam milidetik (ms)
4. **Detail Penyakit & Panduan Penanganan**:
   - Nama penyakit (Bahasa Indonesia & Latin)
   - Patogen penyebab penyakit
   - Ciri-ciri gejala visual
   - Rekomendasi solusi agronomis / pengendalian (kultur teknis, sanitasi, fungisida)
5. **Dashboard Hasil Pengujian Skripsi (Bab 4)**:
   - Tabel ringkasan performa pada data uji (134 sampel citra)
   - *Classification Report* per kelas (Precision, Recall, F1-Score, Support)
   - Grafik komparasi F1-Score ResNet-50 vs ViT-B/16
6. **Ensiklopedia Penyakit Daun Teh**:
   - Katalog 8 kelas penyakit lengkap dengan informasi patogen, gejala, dan solusi.

---

## 🏷️ 8 Kelas Penyakit yang Dikenali

| No | Kelas Dataset | Nama Penyakit | Patogen Penyebab |
|---|---|---|---|
| 1 | `Anthracnose` | Antraknosa | *Colletotrichum theae-sinensis* |
| 2 | `algal leaf` | Bercak Daun Alga | *Cephaleuros virescens* |
| 3 | `bird eye spot` | Bercak Mata Burung | *Cercospora theae* |
| 4 | `brown blight` | Hawar Cokelat | *Glomerella cingulata* |
| 5 | `gray light` | Hawar Abu-abu | *Pestalotiopsis theae* |
| 6 | `healthy` | Daun Sehat | - (Tanaman Sehat) |
| 7 | `red leaf spot` | Bercak Daun Merah | *Phoma theicola* |
| 8 | `white spot` | Bercak Putih | *Phyllosticta theifolia* |

---

## 🚀 Cara Menjalankan Aplikasi

Buka terminal (PowerShell / Command Prompt) di direktori ini, lalu jalankan:

```bash
streamlit run app.py
```

Aplikasi akan otomatis terbuka di browser pada alamat:
`http://localhost:8501`

---

## 📦 Kebutuhan Dependensi

- Python 3.9+
- `streamlit`
- `torch`
- `torchvision`
- `pillow`
- `pandas`
- `numpy`

Untuk menginstal seluruh dependensi:
```bash
pip install -r requirements.txt
```
