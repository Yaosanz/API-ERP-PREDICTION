import pandas as pd

# Membaca data dari file CSV
df = pd.read_csv('dataset_regresi.csv')

# Menghapus kolom 'Is_Expense' jika ada
if 'Is_Expense' in df.columns:
    df = df.drop(columns=['Is_Expense'])

# Menyimpan data ke file baru (opsional)
df.to_csv('dataset_regresi_tanpa_is_expense.csv', index=False)

# Menampilkan 5 baris pertama dari data
print(df.head())
