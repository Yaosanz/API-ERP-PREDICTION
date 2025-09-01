# API Prediksi Keuntungan dengan Flask & TensorFlow untuk Enterpirse Resource Planning

Proyek ini adalah implementasi **REST API** berbasis Flask untuk melakukan **prediksi keuntungan harian** berdasarkan data pemasukan, pengeluaran, kategori transaksi, serta status pengeluaran.  
Model prediksi menggunakan **TensorFlow (Keras)** dengan **scaler** yang telah dilatih sebelumnya, serta memanfaatkan data historis untuk memberikan analisis tambahan (misalnya posisi prediksi dalam distribusi dan statistik kategori).

## Fitur Utama
- Prediksi keuntungan untuk tanggal berikutnya menggunakan model regresi berbasis TensorFlow.
- Preprocessing otomatis pada input (`pemasukan`, `pengeluaran`, `kategori`, `is_expense`).
- Output prediksi dilengkapi dengan:
  - Nilai keuntungan dalam format rupiah.
  - Statistik historis kategori (Q1, Median, Q3, IQR).
  - Posisi prediksi dalam distribusi historis.
  - Persentase perubahan terhadap rata-rata historis.
- Mendukung kategori dinamis sesuai data pelatihan.

## Arsitektur
- **Flask** sebagai backend server.
- **TensorFlow Keras** untuk model prediksi.
- **Joblib** untuk menyimpan dan memuat scaler serta data training.
- **Pandas & NumPy** untuk manipulasi data dan feature engineering.

## Struktur Direktori
```
├── app.py                          # File utama Flask API
├── model\_regresi\_linear\_keras.keras  # Model TensorFlow terlatih
├── scalers\_regresi\_linear.pkl        # Scaler untuk input (X) dan output (y)
├── data\_training\_model.pkl           # Data training historis
├── requirements.txt                  # Daftar dependensi
└── README.md                         # Dokumentasi proyek

````

## Instalasi & Persiapan

1. **Clone repository**
   ```bash
   git clone https://github.com/username/nama-repo.git
   cd nama-repo
````

2. **Buat environment (opsional tetapi direkomendasikan)**

   python -m venv venv
   source venv/bin/activate     # Linux/Mac
   venv\Scripts\activate        # Windows


3. **Install dependensi**

   pip install -r requirements.txt

4. Pastikan file berikut tersedia:

   * model_regresi_linear_keras.keras
   * scalers_regresi_linear.pkl
   * data_training_model.pkl

5. **Menjalankan server**

    ```bash
    api_model.py
````

## Server akan berjalan pada:
http://127.0.0.1:5000

## Endopoint API
http://127.0.0.1:5000/predict

**Request (contoh):**

**URL:**
POST /predict

**Request Body (JSON):**

{
  "pemasukan": 0,
  "pengeluaran": 500000,
  "kategori": "Pengiriman",
  "is_expense": 1
}


**Response (contoh):**

```json
{
  "tanggal_prediksi": "02 September 2025",
  "pemasukan": "Rp0",
  "pengeluaran": "Rp500.000",
  "kategori": "Pengiriman",
  "prediksi_keuntungan": "Rp750.000",
  "rata_rata_historis_kategori": "Rp680.000",
  "selisih": "+Rp70.000",
  "persentase_perubahan": "Naik 10.29% (untung)",
  "status": "naik",
  "posisi_prediksi_dalam_distribusi": "antara median dan Q3",
  "statistik_historis": {
    "Q1": "Rp500.000",
    "Median": "Rp700.000",
    "Q3": "Rp900.000",
    "IQR": "Rp400.000"
  }
}
````

## Dependensi

* Flask
* TensorFlow
* Pandas
* NumPy
* Joblib

Semua dependensi dapat di-install menggunakan:

```bash
pip install -r requirements.txt
````

## Catatan

* API hanya memberikan hasil optimal apabila **file model dan scaler yang sesuai** tersedia.
* Data kategori yang digunakan saat prediksi **harus konsisten** dengan data pada saat model dilatih.
* Endpoint ini dirancang untuk **pengujian dan riset**, bukan untuk skala produksi langsung.
  Untuk deployment skala besar, pertimbangkan penggunaan **Gunicorn** atau **uWSGI** sebagai WSGI server.

## Lisensi

Proyek ini menggunakan lisensi **MIT**.
Silakan gunakan, modifikasi, dan distribusikan sesuai kebutuhan.