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

# --- Format ke Rupiah ---
def to_rupiah(value):
    return f"Rp{int(round(value)):,}".replace(",", ".")

# --- DataFrame untuk Prediksi ---
def buat_dataframe_prediksi(tanggal, pemasukan, pengeluaran, is_expense, kategori):
    df = pd.DataFrame({
        'Tanggal': [tanggal],
        'Total Pemasukan': [pemasukan],
        'Total Pengeluaran': [pengeluaran],
        'Is_Expense': [is_expense],
        'Kategori': [kategori],
    })

    df['Year'] = df['Tanggal'].dt.year
    df['Month'] = df['Tanggal'].dt.month
    df['Day'] = df['Tanggal'].dt.day
    df['DayOfWeek'] = df['Tanggal'].dt.dayofweek
    df['Days_Since_Start'] = (df['Tanggal'] - pd.to_datetime('2021-01-01')).dt.days

    kategori_unik_training = [col.replace('Kategori_', '') for col in data_training.columns if col.startswith('Kategori_')]
    kategori_dummies = pd.get_dummies(df['Kategori'], prefix='Kategori')
    for kategori_nama in kategori_unik_training:
        kolom = f'Kategori_{kategori_nama}'
        if kolom not in kategori_dummies.columns:
            kategori_dummies[kolom] = 0

    df_final = pd.concat([
        df[['Total Pemasukan', 'Total Pengeluaran', 'Is_Expense',
            'Year', 'Month', 'Day', 'DayOfWeek', 'Days_Since_Start']],
        kategori_dummies[[f'Kategori_{k}' for k in kategori_unik_training]]
    ], axis=1)

    if hasattr(scaler_X, 'feature_names_in_'):
        df_final = df_final[scaler_X.feature_names_in_]
    else:
        df_final = df_final.reindex(columns=df_final.columns, fill_value=0)

    return df_final

# --- API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data or 'pemasukan' not in data or 'pengeluaran' not in data or 'kategori' not in data or 'is_expense' not in data:
            return jsonify({'error': 'Masukkan pemasukan, pengeluaran, kategori, dan is_expense'}), 400

        pemasukan = float(data['pemasukan'])
        pengeluaran = float(data['pengeluaran'])
        kategori = str(data['kategori'])
        is_expense = int(data['is_expense'])

        tanggal_besok = datetime.now() + timedelta(days=1)
        df_besok = buat_dataframe_prediksi(tanggal_besok, pemasukan, pengeluaran, is_expense, kategori)

        # --- Prediksi ---
        X_scaled = scaler_X.transform(df_besok)
        y_pred_scaled = model.predict(X_scaled, verbose=0)
        prediksi_besok = scaler_y.inverse_transform(y_pred_scaled)[0][0]

        # --- Statistik Historis ---
        data_kategori_sama = data_training[data_training['Kategori'] == kategori]
        keuntungan = data_kategori_sama['Keuntungan Aktual'].dropna()

        if keuntungan.empty:
            Q1 = Q3 = median = rata_rata_kategori = 0
            posisi_distribusi = "tidak tersedia"
        else:
            Q1 = keuntungan.quantile(0.25)
            Q3 = keuntungan.quantile(0.75)
            median = keuntungan.median()
            rata_rata_kategori = keuntungan.mean()

            if prediksi_besok >= Q3:
                posisi_distribusi = "di atas Q3 (prediksi tinggi)"
            elif prediksi_besok >= median:
                posisi_distribusi = "antara median dan Q3"
            elif prediksi_besok >= Q1:
                posisi_distribusi = "antara Q1 dan median"
            else:
                posisi_distribusi = "di bawah Q1"

        selisih = prediksi_besok - rata_rata_kategori

        if rata_rata_kategori == 0:
            persen = 0
            status = "naik" if prediksi_besok > 0 else "turun" if prediksi_besok < 0 else "tetap"
            persentase_perubahan_label = f"{status.capitalize()} {persen:.2f}%"
        else:
            persen = abs(prediksi_besok - rata_rata_kategori) / abs(rata_rata_kategori) * 100
            persen = min(persen, 100.0)  # batasi maksimal 100%

            if prediksi_besok >= 0 and rata_rata_kategori >= 0:
                status = "naik" if prediksi_besok > rata_rata_kategori else "turun" if prediksi_besok < rata_rata_kategori else "tetap"
                jenis_keuntungan = "untung"
                persentase_perubahan_label = f"{status.capitalize()} {persen:.2f}% ({jenis_keuntungan})"

            elif prediksi_besok < 0 and rata_rata_kategori < 0:
                if abs(prediksi_besok) < abs(rata_rata_kategori):
                    status = "membaik"
                    persentase_perubahan_label = f"Rugi menurun {persen:.2f}%"
                else:
                    status = "memburuk"
                    persentase_perubahan_label = f"Rugi meningkat {persen:.2f}%"

            elif prediksi_besok >= 0 and rata_rata_kategori < 0:
                status = "berbalik untung"
                persentase_perubahan_label = f"Berbalik jadi untung ({persen:.2f}%)"

            elif prediksi_besok < 0 and rata_rata_kategori >= 0:
                status = "berbalik rugi"
                persentase_perubahan_label = f"Berbalik jadi rugi ({persen:.2f}%)"

        # --- Response JSON ---
        response = {
            "tanggal_prediksi": tanggal_besok.strftime("%d %B %Y"),
            "pemasukan": to_rupiah(pemasukan),
            "pengeluaran": to_rupiah(pengeluaran),
            "kategori": kategori,
            "prediksi_keuntungan": to_rupiah(prediksi_besok),
            "rata_rata_historis_kategori": to_rupiah(rata_rata_kategori),
            "selisih": f"{'+' if selisih >= 0 else '-'}{to_rupiah(abs(selisih))}",
            "persentase_perubahan": persentase_perubahan_label,
            "status": status,
            "posisi_prediksi_dalam_distribusi": posisi_distribusi,
            "statistik_historis": {
                "Q1": to_rupiah(Q1),
                "Median": to_rupiah(median),
                "Q3": to_rupiah(Q3),
                "IQR": to_rupiah(Q3 - Q1)
            },
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Run ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)
