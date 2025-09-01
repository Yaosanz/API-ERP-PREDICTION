from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# --- Load Model dan Scaler ---
model = load_model('model_regresi_linear_keras.keras')
scalers = joblib.load('scalers_regresi_linear.pkl')
scaler_X = scalers['scaler_X']
scaler_y = scalers['scaler_y']

# --- Load Data Historis ---
data_training = joblib.load('data_training_model.pkl')

# --- Ambil kolom kategori dan nama ---
kategori_cols_training = [col for col in data_training.columns if col.startswith('Kategori_')]
kategori_names_training = [col.replace('Kategori_', '') for col in kategori_cols_training]

# --- Format ke Rupiah ---
def to_rupiah(value):
    return f"Rp{int(round(value)):,}".replace(",", ".")

# --- Fungsi membuat DataFrame prediksi ---
def buat_df_prediksi(pemasukan, pengeluaran, is_expense, kategori):
    df = pd.DataFrame({
        'Total Pemasukan': [pemasukan],
        'Total Pengeluaran': [pengeluaran],
        'Is_Expense': [is_expense],
    })

    # Dummy kategori sesuai training
    dummies = pd.DataFrame(0, index=[0], columns=kategori_cols_training)
    col_kategori = f'Kategori_{kategori}'
    if col_kategori in dummies.columns:
        dummies[col_kategori] = 1

    df_final = pd.concat([df, dummies], axis=1)

    original_X_cols = [col for col in data_training.columns if col not in ['Keuntungan Aktual', 'Kategori']]
    df_final = df_final.reindex(columns=original_X_cols, fill_value=0)
    return df_final

# --- Fungsi prediksi per kategori ---
def prediksi_kategori(pemasukan, pengeluaran, is_expense, kategori):
    df_pred = buat_df_prediksi(pemasukan, pengeluaran, is_expense, kategori)
    X_scaled = scaler_X.transform(df_pred)
    y_scaled = model.predict(X_scaled, verbose=0)
    return scaler_y.inverse_transform(y_scaled)[0][0]

@app.route('/predict-total', methods=['POST'])
def predict_total():
    try:
        data = request.get_json()
        if not data or not isinstance(data, dict):
            return jsonify({'error': 'Masukkan JSON dengan struktur parameter untuk setiap kategori'}), 400

        total_prediksi = 0
        hasil_detail = []
        tanggal_prediksi = datetime.now() + timedelta(days=1)

        for kategori, params in data.items():
            if kategori not in kategori_names_training:
                return jsonify({'error': f"Kategori '{kategori}' tidak dikenali. Pilih dari: {kategori_names_training}"}), 400

            pemasukan = float(params.get('pemasukan', 0))
            pengeluaran = float(params.get('pengeluaran', 0))
            is_expense = int(params.get('is_expense', 0))

            # Prediksi
            prediksi = prediksi_kategori(pemasukan, pengeluaran, is_expense, kategori)
            total_prediksi += prediksi

            # Statistik historis
            data_kategori = data_training[data_training['Kategori'] == kategori]
            keuntungan_hist = data_kategori['Keuntungan Aktual'].dropna()

            if not keuntungan_hist.empty:
                Q1 = keuntungan_hist.quantile(0.25)
                Q3 = keuntungan_hist.quantile(0.75)
                median = keuntungan_hist.median()
                rata_rata = keuntungan_hist.mean()

                if prediksi >= Q3:
                    posisi = "di atas Q3 (prediksi tinggi)"
                elif prediksi >= median:
                    posisi = "antara median dan Q3"
                elif prediksi >= Q1:
                    posisi = "antara Q1 dan median"
                else:
                    posisi = "di bawah Q1"

                selisih = prediksi - rata_rata
                persen = abs(selisih / rata_rata) * 100 if rata_rata != 0 else 0
                persen = min(persen, 100.0)

                if prediksi >= 0 and rata_rata >= 0:
                    status = "naik" if prediksi > rata_rata else "turun" if prediksi < rata_rata else "tetap"
                    perubahan = f"{status.capitalize()} {persen:.2f}% (untung)"
                elif prediksi < 0 and rata_rata < 0:
                    if abs(prediksi) < abs(rata_rata):
                        perubahan = f"Rugi menurun {persen:.2f}%"
                    else:
                        perubahan = f"Rugi meningkat {persen:.2f}%"
                    status = "membaik" if abs(prediksi) < abs(rata_rata) else "memburuk"
                elif prediksi >= 0 and rata_rata < 0:
                    status = "berbalik untung"
                    perubahan = f"Berbalik jadi untung ({persen:.2f}%)"
                else:
                    status = "berbalik rugi"
                    perubahan = f"Berbalik jadi rugi ({persen:.2f}%)"
            else:
                Q1 = Q3 = median = rata_rata = selisih = persen = 0
                posisi = "tidak tersedia"
                status = "tidak diketahui"
                perubahan = "tidak tersedia"

            # Tambahkan ke hasil
            hasil_detail.append({
                'kategori': kategori,
                'pemasukan': to_rupiah(pemasukan),
                'pengeluaran': to_rupiah(pengeluaran),
                'prediksi_keuntungan': to_rupiah(prediksi),
                'rata_rata_historis_kategori': to_rupiah(rata_rata),
                'selisih': f"{'+' if selisih >= 0 else '-'}{to_rupiah(abs(selisih))}",
                'persentase_perubahan': perubahan,
                'status': status,
                'posisi_prediksi_dalam_distribusi': posisi,
                'statistik_historis': {
                    'Q1': to_rupiah(Q1),
                    'Median': to_rupiah(median),
                    'Q3': to_rupiah(Q3),
                    'IQR': to_rupiah(Q3 - Q1)
                }
            })

        response = {
            'tanggal_prediksi': tanggal_prediksi.strftime('%d %B %Y'),
            'total_prediksi_keuntungan': to_rupiah(total_prediksi),
            'rincian_per_kategori': hasil_detail
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Run ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)