# 🤖 Tugas Besar AI — Prediksi Stunting dengan SVM

Project ini merupakan **Tugas Besar Mata Kuliah Artificial Intelligence (AI)** yang mengimplementasikan algoritma **Support Vector Machine (SVM)** untuk melakukan klasifikasi **stunting pada anak** berdasarkan beberapa fitur kesehatan.

Model diuji menggunakan beberapa kernel SVM yaitu:

* **Linear**
* **Polynomial**
* **RBF**
* **Sigmoid**

Setiap kernel dilakukan **hyperparameter tuning menggunakan GridSearchCV** untuk mendapatkan performa terbaik.

---

# 📌 Project Overview

Repository ini berisi implementasi klasifikasi **stunting** menggunakan **Machine Learning** dengan pendekatan **Support Vector Machine (SVM)**.

Tahapan utama dalam project ini meliputi:

1. Membaca dataset
2. Data preprocessing
3. Encoding data kategorikal
4. Standardisasi fitur
5. Training model SVM
6. Hyperparameter tuning
7. Evaluasi model
8. Visualisasi hasil prediksi
9. Penyimpanan hasil evaluasi ke file CSV dan Excel

Project ini dibuat sebagai implementasi konsep **Artificial Intelligence dalam bidang kesehatan**.

---

# 🚀 Main Features

## 📂 Data Preprocessing

Tahapan preprocessing yang dilakukan:

* Membaca dataset menggunakan **Pandas**
* Menghapus missing values menggunakan `dropna()`
* Encoding data kategorikal menggunakan **LabelEncoder**
* Pemilihan fitur dan label target
* Standardisasi fitur menggunakan **StandardScaler**
* Split dataset menjadi **training data (80%)** dan **testing data (20%)**

---

# 🧠 Machine Learning Model

Algoritma yang digunakan adalah:

**Support Vector Machine (SVM)** dari library **Scikit-learn**

Kernel yang diuji:

* Linear Kernel
* Polynomial Kernel
* Radial Basis Function (RBF)
* Sigmoid Kernel

Setiap kernel dilakukan **Grid Search** untuk mencari kombinasi parameter terbaik.

---

# ⚙️ Hyperparameter Tuning

Hyperparameter tuning dilakukan menggunakan **GridSearchCV**.

### Linear Kernel

Parameter yang diuji:

* `C = [0.1, 1, 10]`

### Polynomial Kernel

Parameter yang diuji:

* `C = [0.1, 1, 10]`
* `degree = [2, 3, 4]`
* `gamma = ['scale', 'auto']`

### RBF Kernel

Parameter yang diuji:

* `C = [0.1, 1, 10]`
* `gamma = ['scale', 'auto']`

### Sigmoid Kernel

Parameter yang diuji:

* `C = [0.1, 1, 10]`
* `gamma = ['scale', 'auto']`

---

# 📊 Model Evaluation

Performa model dievaluasi menggunakan beberapa metrik:

* **Accuracy Score**
* **ROC AUC Score**
* **Classification Report**

Setelah semua kernel diuji, sistem secara otomatis menentukan **kernel terbaik berdasarkan akurasi tertinggi**.

---

# 📈 Visualization

Model menghasilkan visualisasi hasil prediksi menggunakan dua fitur utama:

* **Age**
* **Body Length**

Visualisasi hasil prediksi disimpan sebagai file:

* `hasil_visual_linear_tuned.png`
* `hasil_visual_poly_tuned.png`
* `hasil_visual_rbf_tuned.png`
* `hasil_visual_sigmoid_tuned.png`

Grafik ini menunjukkan **persebaran prediksi stunting vs normal**.

---

# 💾 Output Files

Setelah program dijalankan, beberapa file output akan dihasilkan:

### Evaluation Result

* `hasil_svm_gridsearch.csv`
* `hasil_svm_gridsearch.xlsx`

### Visualization Result

* `hasil_visual_linear_tuned.png`
* `hasil_visual_poly_tuned.png`
* `hasil_visual_rbf_tuned.png`
* `hasil_visual_sigmoid_tuned.png`

---

# 🧰 Tech Stack

| Technology   | Description               |
| ------------ | ------------------------- |
| Python       | Programming Language      |
| Pandas       | Data Processing           |
| NumPy        | Numerical Computation     |
| Matplotlib   | Data Visualization        |
| Seaborn      | Statistical Visualization |
| Scikit-learn | Machine Learning          |

---

# 📁 Project Structure

```
Tubes-AI/
│
├── TubesAI.py
├── stunting_dummy.csv
├── hasil_svm_gridsearch.csv
├── hasil_svm_gridsearch.xlsx
├── hasil_visual_linear_tuned.png
├── hasil_visual_poly_tuned.png
├── hasil_visual_rbf_tuned.png
├── hasil_visual_sigmoid_tuned.png
└── README.md
```

---

# ⚙️ Requirements

Pastikan sudah menginstall:

* Python 3.x
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* openpyxl

---

# 🛠 Installation

Clone repository:

```
git clone https://github.com/username/tubes-ai.git
```

Masuk ke folder project:

```
cd tubes-ai
```

Install dependencies:

```
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

---

# ▶️ Run Program

Jalankan program dengan perintah:

```
python TubesAI.py
```

---

# 🧠 Workflow Program

Alur kerja program:

1. Membaca dataset `stunting_dummy.csv`
2. Melakukan preprocessing data
3. Encoding fitur kategorikal
4. Standardisasi fitur
5. Membagi dataset menjadi train dan test
6. Melakukan GridSearchCV untuk setiap kernel
7. Mengevaluasi performa model
8. Menyimpan hasil evaluasi ke file CSV dan Excel
9. Menyimpan visualisasi hasil prediksi
10. Menentukan kernel terbaik secara otomatis

---

# 📚 Learning Outcomes

Project ini membantu memahami:

* konsep **machine learning classification**
* implementasi **Support Vector Machine**
* penggunaan **GridSearchCV untuk hyperparameter tuning**
* evaluasi model menggunakan berbagai metrik
* visualisasi hasil klasifikasi
* penerapan AI pada studi kasus kesehatan

---

# 🎯 Project Purpose

Project ini dibuat sebagai:

* **Tugas Besar Mata Kuliah Artificial Intelligence**
* implementasi algoritma machine learning
* latihan preprocessing, training, tuning, dan evaluasi model
* portfolio project di bidang **Artificial Intelligence dan Machine Learning**

---

# 👨‍💻 Author

**Mukhamad Sofyan**
GitHub: https://github.com/mukhamadsofyan

---

# 🤝 Contributors

Project ini dikembangkan oleh:

* **Mukhamad Sofyan**
* **Moh. Ravlindo Saputra**

---

# 📄 License

Project ini dibuat untuk **tujuan pembelajaran dan portfolio akademik**.
