import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# ===== BACA DATASET =====
csv_path = 'stunting_dummy.csv'  # Ganti jika file ada di folder lain
df = pd.read_csv(csv_path)

# ===== PRA-PROSES DATA =====
df.dropna(inplace=True)
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Asi Ekslusif'] = le.fit_transform(df['Asi Ekslusif'])
df['Outcome Stunting'] = le.fit_transform(df['Outcome Stunting'])

# ===== FITUR & LABEL =====
X = df[['Sex', 'Age', 'Birth Weight', 'Birth Length', 'Body Weight', 'Body Length', 'Asi Ekslusif']]
y = df['Outcome Stunting']
label_names = ['Normal', 'Stunting']

# Simpan nama kolom untuk visualisasi fitur tertentu
fitur_x = 'Age'
fitur_y = 'Body Length'
fitur_x_idx = X.columns.get_loc(fitur_x)
fitur_y_idx = X.columns.get_loc(fitur_y)

# ===== STANDARISASI & SPLIT =====
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ===== DEFINISI KERNEL DAN PARAMETER GRID =====
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
param_grids = {
    'linear': {'C': [0.1, 1, 10]},
    'poly': {'C': [0.1, 1, 10], 'degree': [2, 3, 4], 'gamma': ['scale', 'auto']},
    'rbf': {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']},
    'sigmoid': {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
}

results = []

# ===== TUNING, EVALUASI, VISUALISASI =====
for kernel in kernels:
    print(f"\n Tuning dan Evaluasi Kernel: {kernel.upper()}")

    svc = SVC(kernel=kernel, probability=True)
    grid = GridSearchCV(svc, param_grids[kernel], cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_params = grid.best_params_

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f" Best Params: {best_params}")
    print(f"Akurasi : {acc * 100:.2f}%")
    print(f"AUC     : {auc:.3f}")
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_names))

    results.append({
        "Kernel": kernel.upper(),
        "Akurasi (%)": round(acc * 100, 2),
        "AUC": round(auc, 3),
        "Best Params": str(best_params)
    })

    # ===== VISUALISASI DENGAN 2 FITUR: Age vs Body Length =====
    X_vis = X_test[:, [fitur_x_idx, fitur_y_idx]]
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=X_vis[:, 0], y=X_vis[:, 1], hue=y_pred, palette='coolwarm', alpha=0.7)
    plt.title(f"Prediksi SVM ({kernel.upper()} Kernel)\nBest Params: {best_params}")
    plt.xlabel(f"{fitur_x} (standar)")
    plt.ylabel(f"{fitur_y} (standar)")
    plt.legend(title='Prediksi', labels=label_names)
    plt.tight_layout()
    plt.savefig(f'hasil_visual_{kernel}_tuned.png')
    plt.close()

# ===== RINGKASAN HASIL & SIMPAN =====
summary_df = pd.DataFrame(results)
print("\n📊 RINGKASAN HASIL GRIDSEARCHCV SVM:")
print(summary_df)

summary_df.to_csv('hasil_svm_gridsearch.csv', index=False)
summary_df.to_excel('hasil_svm_gridsearch.xlsx', index=False)
print("\n Hasil disimpan ke 'hasil_svm_gridsearch.csv' dan 'hasil_svm_gridsearch.xlsx'.")

# ===== KESIMPULAN TERBAIK =====
best = summary_df.loc[summary_df['Akurasi (%)'].idxmax()]
print("\n KESIMPULAN OTOMATIS:")
print(f"Kernel terbaik adalah **{best['Kernel']}**, dengan Akurasi {best['Akurasi (%)']}% dan AUC {best['AUC']}.\nBest Hyperparameters: {best['Best Params']}")
