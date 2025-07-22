import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===== KONFIGURASI HALAMAN =====
st.set_page_config(
    page_title="Dashboard & Prediksi Kesehatan",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== FUNGSI UNTUK MEMUAT DAN MEMBERSIHKAN DATA =====
@st.cache_data
def load_and_clean_data():
    file_path = 'healthcare_dataseet.csv'
    if not os.path.exists(file_path):
        st.error(f"❌ File dataset '{file_path}' tidak ditemukan. Pastikan file berada di folder yang sama dengan skrip Anda.")
        st.stop()

    df = pd.read_csv(file_path, delimiter=';', header=0)
    
    # Ganti nama kolom secara otomatis
    keyword_mapping = {
        'Age': 'Age', 'Gender': 'Gender', 'Height': 'Height_(cm)',
        'Weight': 'Weight_(kg)', 'BMI': 'BMI'
    }
    df.rename(columns=keyword_mapping, inplace=True, errors='ignore')
        
    required_cols = ['Age', 'Gender', 'Height (cm)', 'Weight (kg)', 'BMI']
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Kolom yang dibutuhkan '{col}' tidak ditemukan di dalam file CSV Anda.")
            st.stop()

    df_cleaned = df[required_cols].copy()

    # Konversi tipe data dan tangani missing values
    for col in ['Age', 'Height (cm)', 'Weight (kg)', 'BMI']:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col].astype(str).str.replace(',', '.'), errors='coerce')
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
    
    df_cleaned.dropna(inplace=True)
    df_cleaned['Age'] = df_cleaned['Age'].astype(int)
    
    # Buat Kategori BMI
    bins_bmi = [0, 18.5, 25, 30, np.inf]
    labels_bmi = ['Underweight', 'Normal', 'Overweight', 'Obese']
    df_cleaned['BMI_Category'] = pd.cut(df_cleaned['BMI'], bins=bins_bmi, labels=labels_bmi, right=False)
    
    df_cleaned.dropna(subset=['BMI_Category'], inplace=True)
    return df_cleaned

# ===== FUNGSI UNTUK MELATIH MODEL PREDIKSI =====
@st.cache_resource
def train_model(df):
    features = ['Age', 'Gender', 'Height (cm)', 'Weight (kg)']
    target = 'BMI_Category'
    X = df[features]
    y = df[target]

    X_encoded = pd.get_dummies(X, columns=['Gender'], drop_first=True)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    trained_columns = X_encoded.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    return model, le, trained_columns, accuracy, report, cm

# Muat data dan latih model
df_cleaned = load_and_clean_data()
model, le, trained_columns, accuracy, report, cm = train_model(df_cleaned)

# ===== SIDEBAR =====
st.sidebar.title("⚙️ Filter Interaktif")
st.sidebar.markdown("Filter ini akan memengaruhi data pada tab 'Visualisasi & Analisis' dan 'Tampilan Data'.")

selected_gender = st.sidebar.multiselect("Pilih Gender", options=df_cleaned['Gender'].unique(), default=df_cleaned['Gender'].unique())
min_age, max_age = int(df_cleaned['Age'].min()), int(df_cleaned['Age'].max())
selected_age = st.sidebar.slider("Pilih Rentang Usia", min_value=min_age, max_value=max_age, value=(min_age, max_age))
df_filtered = df_cleaned[(df_cleaned['Gender'].isin(selected_gender)) & (df_cleaned['Age'] >= selected_age[0]) & (df_cleaned['Age'] <= selected_age[1])]


# ===== JUDUL UTAMA =====
st.title("🏥 Dashboard Analisis & Prediksi Kesehatan")

# ===== TABS =====
tab1, tab2, tab3 = st.tabs(["**📈 Visualisasi & Analisis**", "**🤖 Prediksi BMI**", "**📄 Tampilan Data**"])

# --- TAB 1: VISUALISASI & ANALISIS ---
with tab1:
    st.header("Analisis Data Terfilter")
    
    if df_filtered.empty:
        st.warning("Tidak ada data yang cocok dengan filter yang Anda pilih. Silakan sesuaikan filter di sidebar.")
    else:
        st.success(f"Menampilkan **{len(df_filtered)}** baris data sesuai filter Anda.")
        
        st.subheader("Jumlah Data per Kategori BMI")
        bmi_counts = df_filtered['BMI_Category'].value_counts().sort_index()
        fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
        sns.barplot(x=bmi_counts.index, y=bmi_counts.values, ax=ax_bar, palette='viridis')
        ax_bar.set_title("Jumlah Data per Kategori BMI")
        ax_bar.set_xlabel("Kategori BMI")
        ax_bar.set_ylabel("Jumlah")
        st.pyplot(fig_bar)
        
        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Distribusi Kategori BMI (%)")
            bmi_pie_counts = df_filtered['BMI_Category'].value_counts()
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.pie(bmi_pie_counts, labels=bmi_pie_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
            ax.axis('equal')
            st.pyplot(fig)
        with col2:
            st.subheader("Distribusi Usia")
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.histplot(df_filtered['Age'], bins=20, kde=True, ax=ax, color='skyblue')
            st.pyplot(fig)
            
        st.divider()
        
        st.subheader("Tren Rata-Rata BMI Berdasarkan Usia")
        avg_bmi_by_age = df_filtered.groupby('Age')['BMI'].mean().reset_index()
        fig_line, ax_line = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=avg_bmi_by_age, x='Age', y='BMI', marker='o', ax=ax_line, color='green')
        ax_line.set_title("Tren Rata-Rata BMI untuk Data Terfilter")
        ax_line.set_xlabel("Usia")
        ax_line.set_ylabel("Rata-Rata BMI")
        ax_line.grid(True)
        st.pyplot(fig_line)

# --- TAB 2: PREDIKSI BMI ---
with tab2:
    st.header("🤖 Prediksi Kategori BMI Anda")
    st.markdown("Masukkan data Anda untuk mendapatkan prediksi.")
    st.info(f"**Akurasi Model Prediksi:** **{accuracy:.2%}**")

    with st.expander("Lihat Detail Evaluasi Model 📜"):
        st.subheader("Laporan Klasifikasi (Classification Report)")
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

        st.subheader("Matriks Kebingungan (Confusion Matrix)")
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=le.classes_, yticklabels=le.classes_, ax=ax_cm)
        ax_cm.set_xlabel("Prediksi")
        ax_cm.set_ylabel("Aktual")
        st.pyplot(fig_cm)
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Usia (Tahun)", min_value=1, max_value=120, value=30, step=1)
            height = st.number_input("Tinggi Badan (cm)", min_value=50.0, max_value=250.0, value=165.0, step=0.5)
        with col2:
            gender_options = list(df_cleaned['Gender'].unique())
            gender = st.selectbox("Gender", options=gender_options)
            weight = st.number_input("Berat Badan (kg)", min_value=10.0, max_value=300.0, value=60.0, step=0.5)
        submit_button = st.form_submit_button(label="🔮 Prediksi Sekarang")

    if submit_button:
        input_data = pd.DataFrame({'Age': [age], 'Gender': [gender], 'Height_cm': [height], 'Weight_kg': [weight]})
        input_encoded = pd.get_dummies(input_data, columns=['Gender'])
        input_final = input_encoded.reindex(columns=trained_columns, fill_value=0)
        prediction_encoded = model.predict(input_final)
        prediction_label = le.inverse_transform(prediction_encoded)[0]
        st.success(f"**Hasil Prediksi: Kategori BMI Anda adalah `{prediction_label}`** 🎉")

# --- TAB 3: TAMPILAN DATA ---
with tab3:
    st.header("📄 Tampilan Data Terfilter")
    st.markdown("Berikut adalah tabel data berdasarkan filter yang Anda pilih di sidebar.")
    if df_filtered.empty:
        st.warning("Tidak ada data yang cocok dengan filter yang Anda pilih.")
    else:
        st.dataframe(df_filtered)