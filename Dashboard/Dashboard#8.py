import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ------------------------------
# 📌 Judul Aplikasi
st.title("🌤️ Analisis Polusi Udara - Kota Wanliu 🌧️")

# ------------------------------
# Membaca File
@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), "PRSA_Data_Wanliu_20130301-20170228.csv")
    Wanliu_df = pd.read_csv(file_path)
    Wanliu_df['date'] = pd.to_datetime(Wanliu_df[['year', 'month', 'day']])
    Wanliu_df['year_only'] = Wanliu_df['date'].dt.year
    return Wanliu_df

Wanliu_df = load_data()

# ------------------------------
# Sidebar Filter Tahun
st.sidebar.header("🔍 Pilih Tahun")
tahun = Wanliu_df['year_only'].unique()
tahun_dipilih = st.sidebar.selectbox("Pilih Tahun", tahun)

# Filter data berdasarkan tahun yang dipilih
Wanliu_df['date'] = pd.to_datetime(Wanliu_df[['year', 'month', 'day', 'hour']])
Wanliu_df['quarter'] = Wanliu_df['date'].dt.quarter
Wanliu_df['year_only'] = Wanliu_df['date'].dt.year

filtered_Wanliu = Wanliu_df[Wanliu_df['year_only'] == tahun_dipilih].copy()

# ------------------------------
# Menampilkan DataFrame
st.subheader("📌 DataFrame yang Dipilih")

# Membuat expander untuk menampilkan data
with st.expander(" **DataFrame**", expanded=False):
    st.dataframe(filtered_Wanliu)

# Statistik Deskriptif
st.subheader("📌 Statistik Deskriptif")

# Membuat expander
with st.expander("**Rangkuman Parameter Statistik**", expanded=False):
    st.write(filtered_Wanliu.describe())

# ------------------------------
# Exploratory Data Analysis (EDA)
st.subheader("📌 Exploratory Data Analysis (EDA)")

kolom_numerik_eda = ['PM2.5', 'PM10', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
# Deteksi Outlier dengan IQR
Q1_eda = filtered_Wanliu[kolom_numerik_eda].quantile(0.25)
Q3_eda = filtered_Wanliu[kolom_numerik_eda].quantile(0.75)
IQR_eda = Q3_eda - Q1_eda

batas_bawah = Q1_eda - 1.5 * IQR_eda
batas_atas = Q3_eda + 1.5 * IQR_eda

clean_df = filtered_Wanliu[~((filtered_Wanliu[kolom_numerik_eda] < batas_bawah) | (filtered_Wanliu[kolom_numerik_eda] > batas_atas)).any(axis=1)]

# Cek jumlah data sebelum dan sesudah
st.write(f"Jumlah data sebelum pembersihan: {filtered_Wanliu.shape[0]}")
st.write(f"Jumlah data setelah pembersihan: {clean_df.shape[0]}")

with st.expander("**DataFrame yang Sudah Dibersihkan**", expanded=False):
    st.write(clean_df.describe())

# Visualisasi Distribusi Data Kotor
st.write("### Distribusi Data")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
cols = ['PM2.5', 'PM10', 'TEMP', 'PRES', 'DEWP', 'WSPM']
for ax, col in zip(axes.flatten(), cols):
    sns.histplot(filtered_Wanliu[col], bins=30, kde=True, ax=ax)
    ax.set_title(f'Distribusi {col}')
plt.tight_layout()

# Membuat expander data kotor
with st.expander("### Distribusi Data Sebelum Pembersihan", expanded=False):
    st.pyplot(fig)

# Visualisasi Distribusi Data Bersih
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
cols = ['PM2.5', 'PM10', 'TEMP', 'PRES', 'DEWP', 'WSPM']
for ax, col in zip(axes.flatten(), cols):
    sns.histplot(clean_df[col], bins=30, kde=True, ax=ax)
    ax.set_title(f'Distribusi {col}')
plt.tight_layout()

# Membuat expander data bersih
with st.expander("### Distribusi Data Setelah Pembersihan", expanded=False):
    st.pyplot(fig)

# Boxplot untuk Outlier
st.write("### Deteksi Outlier dengan Boxplot")
# Boxplot Data Kotor
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=filtered_Wanliu[cols])
plt.xticks(rotation=45)
plt.title('Boxplot untuk Mendeteksi Outlier pada Data Sebelum Pembersihan')

with st.expander("### Deteksi Outlier dengan Boxplot pada Data Sebelum Pembersihan",expanded=False):
    st.pyplot(fig)

# Boxplot Data Bersih
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=clean_df[cols])
plt.xticks(rotation=45)
plt.title('Boxplot untuk Mendeteksi Outlier pada Data Sesudah Pembersihan')

with st.expander("### Deteksi Outlier dengan Boxplot pada Data Sesudah Pembersihan",expanded=False):
    st.pyplot(fig)

avg_pm10_per_quarter_year = clean_df.groupby(['year_only', 'quarter'])['PM10'].mean().reset_index()
# Filter data berdasarkan tahun yang dipilih
data = avg_pm10_per_quarter_year[avg_pm10_per_quarter_year['year_only'] == tahun_dipilih]

# ------------------------------
# Visualization dan Explanatory Analysis
st.subheader("📌 Visualization dan Explanatory Analysis")
# ------------------------------
st.subheader(f"📈 Rata-Rata PM10 pada Tahun {tahun_dipilih}")

plt.title(f"Tren Rata-Rata PM10 per Quarter pada Tahun {tahun_dipilih}")
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(
    x='quarter', y='PM10',
    data=data, marker='o', ax=ax
)
ax.set_title(f'Tren Rata-rata PM10 per Quarter pada Tahun {tahun_dipilih}')
ax.set_xlabel('Quarter')
ax.set_ylabel('Rata-rata PM10 (μg/m³)')
ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
ax.grid(True, linestyle='--')
st.pyplot(fig)

# fig, ax = plt.subplots(figsize=(12, 6))
# sns.lineplot(
#     x='quarter', y='PM10', hue='year_only',
#     data=filtered_data, marker='o', ax=ax
# )
# ax.set_title('Rata-rata PM10 per Quarter')
# ax.set_xlabel('Quarter')
# ax.set_ylabel('Rata-rata PM10 (μg/m³)')
# ax.set_xticks([1, 2, 3, 4])
# ax.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
# ax.legend(title='Tahun')
# ax.grid(True, linestyle='--')
# st.pyplot(fig)

with st.expander("### Analisis Tren Rata-rata PM10",expanded=False):
    st.write("""Berdasarkan hasil tren, terdapat variasi rata-rata PM10
    antar kuartal dalam setiap tahun sehingga dapat diidentifikasi adanya 
    faktor musiman yang mempengaruhi konsentrasi PM10.Meskipun memiliki 
    variasi, dalam beberapa tahun dapat menunjukkan puncak PM10 pada kuartal
    tertentu. Rata-rata PM10 memiliki konsentrasi tertinggi di Q1 kemudian 
    menurun sampai Q3 dan kembali naik di pada Q4.
""")

# ------------------------------
# Heatmap Korelasi
st.subheader("🔥 Heatmap Korelasi")
fig, ax = plt.subplots(figsize=(10, 6))
corr = clean_df[cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
st.pyplot(fig)

with st.expander(" **Kesimpulan**", expanded=False):
    st.write("""Pengguna dapat menyesuaikan bulan untuk melihat pola polusi berdasarkan musim.
    Berdasarkan heatmap yang dihasilkan dari perhitungan, variabel TEMP (suhu udara) 
    memberikan nilai korelasi negatif yang artinya ketika suhu meningkat, 
    kadar PM2.5 cenderung menurun. Variabel PRES (tekanan udara) dan DEWP (titik embun) 
    tidak memberikan pengaruh besar karena nilainya yang sangat kecil. 
    Variabel RAIN(hujan), memiliki nilai korelasi negatif , sehingga ketika terjadi 
    hujan polusi udara akan menurun. Variabel WSPM(kecepatan angin) memiliki pengaruh 
    terhadap penyebaran polutan PM2.5 ini, semakin besar nilainya atau tinggi kecepatan 
    anginnya konsentrasi PM2.5 semakin rendah polutan akan tersebar luas.Heatmap korelasi 
    memperlihatkan hubungan antara PM2.5, PM10, dan faktor lainnya secara lebih interaktif.
""")