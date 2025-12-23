import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load Data
# Pastikan path ini sesuai dengan letak csv Anda
print("Memuat data...")
data = pd.read_csv("fraud_dataset_preprocessing/train_processed.csv")

# Pisahkan Fitur (X) dan Target (y)
# Sesuaikan 'IsFraud' dengan nama kolom target Anda jika berbeda
X = data.drop("IsFraud", axis=1)
y = data["IsFraud"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Setup MLflow dengan Autolog
# Autolog akan otomatis mencatat parameter default model dan metrik akurasi
mlflow.set_experiment("Fraud_Detection_Basic")
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="Basic_Training_No_Tuning"):
    print("Memulai training model...")
    
    # 3. Train Model (Tanpa Tuning)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Evaluasi Sederhana
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    
    print(f"Training selesai. Akurasi: {acc}")
    print("Semua metrik dan model telah tercatat otomatis oleh MLflow Autolog.")