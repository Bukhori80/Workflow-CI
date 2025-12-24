import pandas as pd
import sys
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score

# ==========================================
# 1. MENANGKAP PARAMETER DARI TERMINAL (PENTING!)
# ==========================================
# Logika: Jika ada input dari luar (via mlflow run), pakai itu.
# Jika tidak ada, pakai nilai default (100 dan 5).
if len(sys.argv) > 1:
    n_estimators = int(sys.argv[1])
    max_depth = int(sys.argv[2])
else:
    n_estimators = 100
    max_depth = 5

print(f" Memulai Training Model...")
print(f" Parameter: n_estimators={n_estimators}, max_depth={max_depth}")

# ==========================================
# 2. LOAD & PREPARE DATASET
# ==========================================
# Pastikan nama file CSV sesuai dengan yang ada di folder Anda
try:
    # Ganti 'fraud_dataset.csv' jika nama file Anda berbeda
    # Jika dataset ada di dalam folder 'fraud_dataset', gunakan: 'fraud_dataset/nama_file.csv'
    df = pd.read_csv('fraud_dataset_preprocessing/train_processed.csv')
    
    # --- TAMBAHAN BARU: BERSIHKAN NAMA KOLOM ---
    # Ubah semua nama kolom jadi huruf kecil & hapus spasi
    df.columns = df.columns.str.lower().str.strip()
    
    print("Nama Kolom yang ditemukan:", df.columns.tolist()) # Buat ngecek di terminal
    # -------------------------------------------
    
    df = df.fillna(0)
    
    # Sekarang pasti aman pakai 'fraud' (huruf kecil)
    X = df.drop('isfraud', axis=1)
    y = df['isfraud']
    
except FileNotFoundError:
    print(" Error: File CSV tidak ditemukan! Pastikan nama file di kodingan sesuai.")
    sys.exit()

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 3. MULAI MLFLOW RUN
# ==========================================
# Tetapkan nama eksperimen
experiment_name = "Credit_Scoring_Experiment"

# Logika otomatis: Cek eksperimen, jika belum ada maka buat baru
try:
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
except Exception as e:
    print(f"Info: Menggunakan default experiment karena: {e}")

with mlflow.start_run():
    # A. LOG PARAMETER (Agar terekam settingan apa yang dipakai)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("model_type", "RandomForest")

    # B. TRAINING MODEL
    model = RandomForestClassifier(n_estimators=n_estimators, 
                                   max_depth=max_depth, 
                                   class_weight='balanced',  
                                   random_state=42)
    model.fit(X_train, y_train)

    # C. PREDIKSI & EVALUASI
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    recall = recall_score(y_test, predictions)

    # D. LOG METRICS (Agar hasil ujian terekam)
    print(f" Hasil -> Accuracy: {accuracy:.4f} | F1: {f1:.4f} | Recall: {recall:.4f}")
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("recall_score", recall)

    # E. LOG MODEL (Simpan file model .pkl ke dalam MLflow)
    mlflow.sklearn.log_model(model, "model")
    
    print(" Selesai! Model dan metrik berhasil disimpan ke MLflow.")