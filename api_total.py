from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# --- Load model, scaler, dan data training ---
model = load_model('model_regresi_linear_keras.keras')
scalers = joblib.load('scalers_regresi_linear.pkl')
scaler_X = scalers['scaler_X']
scaler_y = scalers['scaler_y']
data_training = joblib.load('data_training_model.pkl')

# --- Load label encoder jika ada ---
try:
    le_kategori = joblib.load('label_encoder_Kategori.pkl')
    label_encoder_loaded = True
except:
    le_kategori = None
    label_encoder_loaded = False

# --- Ambil kolom dan nama kategori ---
kategori_cols_training = [col for col in data_training.columns if col.startswith('Kategori_')]
kategori_names_training = (
    [col.replace('Kategori_', '') for col in kategori_cols_training]
    if kategori_cols_training else
    (le_kategori.classes_.tolist() if label_encoder_loaded else [])
)

# --- Format ke Rupiah ---
def to_rupiah(value):
    return f"Rp{int(round(value)):,}".replace(",", ".")

# --- Buat DataFrame prediksi ---
def buat_df_prediksi(pemasukan, pengeluaran, is_expense, kategori):
    df = pd.DataFrame({
        'Total Pemasukan': [pemasukan],
        'Total Pengeluaran': [pengeluaran],
        'Is_Expense': [is_expense]
    })

    if kategori_cols_training:
        dummies = pd.DataFrame(0, index=[0], columns=kategori_cols_training)
        col_kategori = f'Kategori_{kategori}'
        if col_kategori not in dummies.columns:
            return None
        dummies[col_kategori] = 1
        df_final = pd.concat([df, dummies], axis=1)
    elif label_encoder_loaded:
        try:
            df['Kategori'] = le_kategori.transform([kategori])
        except:
            return None
        df_final = df
    else:
        return None

    original_X_cols = [col for col in data_training.columns if col not in ['Keuntungan Aktual']]
    df_final = df_final.reindex(columns=original_X_cols, fill_value=0)
    return df_final

# --- Prediksi dan analisis ---
def prediksi_kategori(pemasukan, pengeluaran, is_expense, kategori):
    df_pred = buat_df_prediksi(pemasukan, pengeluaran, is_expense, kategori)
    if df_pred is None:
        return None

    X_scaled = scaler_X.transform(df_pred)
    y_scaled = model.predict(X_scaled, verbose=0)
    prediksi = scaler_y.inverse_transform(y_scaled)[0][0]

    # Ambil historis keuntungan
    if kategori_cols_training:
        keuntungan_hist = data_training[data_training[f'Kategori_{kategori}'] == 1]['Keuntungan Aktual']
    elif label_encoder_loaded:
        encoded = le_kategori.transform([kategori])[0]
        keuntungan_hist = data_training[data_training['Kategori'] == encoded]['Keuntungan Aktual']
    else:
        keuntungan_hist = pd.Series()

    if keuntungan_hist.empty:
        return {
            'prediksi': prediksi,
            'rata_rata': 0,
            'selisih': 0,
            'persen': 0,
            'status': 'tidak diketahui',
            'posisi': 'tidak tersedia',
            'statistik_historis': {'Q1': 0, 'Median': 0, 'Q3': 0, 'IQR': 0}
        }

    rata_rata = keuntungan_hist.mean()
    Q1 = keuntungan_hist.quantile(0.25)
    Q3 = keuntungan_hist.quantile(0.75)
    median = keuntungan_hist.median()

    selisih = prediksi - rata_rata
    persen = abs(selisih / rata_rata) * 100 if rata_rata != 0 else 0
    persen = min(persen, 100)

    posisi = (
        "di atas Q3 (prediksi tinggi)" if prediksi >= Q3 else
        "antara median dan Q3" if prediksi >= median else
        "antara Q1 dan median" if prediksi >= Q1 else
        "di bawah Q1"
    )

    if prediksi >= 0 and rata_rata >= 0:
        status = "naik" if prediksi > rata_rata else "turun" if prediksi < rata_rata else "tetap"
        perubahan = f"{status.capitalize()} {persen:.2f}% (untung)"
    elif prediksi < 0 and rata_rata < 0:
        status = "membaik" if abs(prediksi) < abs(rata_rata) else "memburuk"
        perubahan = f"Rugi {'menurun' if status == 'membaik' else 'meningkat'} {persen:.2f}%"
    elif prediksi >= 0 and rata_rata < 0:
        status = "berbalik untung"
        perubahan = f"Berbalik jadi untung ({persen:.2f}%)"
    else:
        status = "berbalik rugi"
        perubahan = f"Berbalik jadi rugi ({persen:.2f}%)"

    return {
        'prediksi': prediksi,
        'rata_rata': rata_rata,
        'selisih': selisih,
        'persen': persen,
        'status': status,
        'perubahan': perubahan,
        'posisi': posisi,
        'statistik_historis': {
            'Q1': Q1,
            'Median': median,
            'Q3': Q3,
            'IQR': Q3 - Q1
        }
    }

@app.route('/predict-total', methods=['POST'])
def predict_total():
    try:
        data = request.get_json()
        if not isinstance(data, dict):
            return jsonify({'error': 'Data input harus dalam format JSON dictionary'}), 400

        total_prediksi = 0
        hasil_detail = []
        tanggal_prediksi = datetime.now() + timedelta(days=1)

        for kategori, params in data.items():
            if kategori not in kategori_names_training:
                return jsonify({'error': f"Kategori '{kategori}' tidak dikenali. Pilihan: {kategori_names_training}"}), 400

            pemasukan = float(params.get('pemasukan', 0))
            pengeluaran = float(params.get('pengeluaran', 0))
            is_expense = int(params.get('is_expense', 0))

            hasil = prediksi_kategori(pemasukan, pengeluaran, is_expense, kategori)
            if hasil is None:
                return jsonify({'error': f"Kategori '{kategori}' tidak dapat diproses"}), 400

            total_prediksi += hasil['prediksi']

            hasil_detail.append({
                'kategori': kategori,
                'pemasukan': to_rupiah(pemasukan),
                'pengeluaran': to_rupiah(pengeluaran),
                'prediksi_keuntungan': to_rupiah(hasil['prediksi']),
                'rata_rata_historis_kategori': to_rupiah(hasil['rata_rata']),
                'selisih': f"{'+' if hasil['selisih'] >= 0 else '-'}{to_rupiah(abs(hasil['selisih']))}",
                'persentase_perubahan': hasil['perubahan'],
                'status': hasil['status'],
                'posisi_prediksi_dalam_distribusi': hasil['posisi'],
                'statistik_historis': {
                    'Q1': to_rupiah(hasil['statistik_historis']['Q1']),
                    'Median': to_rupiah(hasil['statistik_historis']['Median']),
                    'Q3': to_rupiah(hasil['statistik_historis']['Q3']),
                    'IQR': to_rupiah(hasil['statistik_historis']['IQR'])
                }
            })

        return jsonify({
            'tanggal_prediksi': tanggal_prediksi.strftime('%d %B %Y'),
            'total_prediksi_keuntungan': to_rupiah(total_prediksi),
            'rincian_per_kategori': hasil_detail
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
