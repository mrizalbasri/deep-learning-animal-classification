# Proyek Klasifikasi Gambar: Animals-10 Image Classification (Butterfly, Chicken, Horse, Spider)

Submission Akhir Kelas **Belajar Pengembangan Machine Learning (DLP)** - Dicoding Indonesia.

- **Nama:** M. Rizal Basri
- **Email:** rizalbasri800@gmail.com
- **ID Dicoding:** rizalbasri

---

## 📌 Ringkasan Proyek

Proyek ini bertujuan untuk membangun model deep learning berbasis Convolutional Neural Network (**CNN**) dengan Keras `Sequential` untuk mengklasifikasikan citra hewan ke dalam 4 kategori kelas (*Butterfly*, *Chicken*, *Horse*, *Spider*) secara akurat dan efisien, serta mengonversinya ke dalam berbagai format deployment (**SavedModel**, **TF-Lite**, dan **TensorFlow.js**).

Seluruh tahapan mulai dari pengunduhan dataset, eksplorasi variasi resolusi citra, pembagian dataset (Train 80%, Validation 10%, Test 10%), augmentasi citra, arsitektur CNN Sequential, callback penghenti training otomatis, visualisasi akurasi & loss, confusion matrix, classification report, ekspor multi-format model, hingga inferensi model TF-Lite telah dijalankan secara berurutan dan terverifikasi 100%.

Ukuran paket berkas submission (`submission.zip`) dibuat ringkas (~7 MB) sehingga berada jauh di bawah batas maksimal 25 MB di Dicoding.

---

## 🌟 Pemenuhan Kriteria Submission

| Kriteria / Saran | Target Dicoding | Implementasi Proyek | Status |
|---|---|---|:---:|
| **Kriteria 1** | Minimal 1.000 citra (Saran: >= 10.000 citra) | Menggunakan **12.654 citra** (> 10.000 citra memenuhi kriteria saran bintang 5) | ✅ Terpenuhi |
| **Kriteria 2** | Bebas & Tidak menggunakan dataset kelas (Bukan Rock-Paper-Scissors / X-Ray) | Dataset Animals-10 (Butterfly, Chicken, Horse, Spider) | ✅ Terpenuhi |
| **Kriteria 3** | Dataset dibagi menjadi Train, Validation, dan Test | 80% Train (10.121 citra), 10% Validation (1.264 citra), 10% Test (1.269 citra) dengan kode pembagian dataset di notebook | ✅ Terpenuhi |
| **Kriteria 4** | Model Sequential, Conv2D, Pooling Layer | `Sequential` dengan MobileNetV2 feature extractor, `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`, `Dropout` | ✅ Terpenuhi |
| **Kriteria 5** | Akurasi pada Training dan Testing Set minimal 85% | Akurasi Training dan Testing Set mencapai **> 90%** | ✅ Terpenuhi |
| **Kriteria 6** | Plot visualisasi Akurasi dan Loss | Grafik akurasi & loss training vs validasi lengkap dengan legend dan grid | ✅ Terpenuhi |
| **Kriteria 7** | Export ke SavedModel, TF-Lite, dan TFJS | Format SavedModel, TF-Lite (`model.tflite` & `label.txt`), TFJS (`model.json` & shard) | ✅ Terpenuhi |
| **Kriteria Tambahan** | Ukuran berkas submission < 25 MB | Ukuran `submission.zip` **hanya ~7 MB** (langsung bisa di-upload tanpa Google Drive) | ✅ Terpenuhi |
| **Saran 1** | Mengimplementasikan Callback | `EarlyStopping`, `ReduceLROnPlateau`, dan custom `TargetAccuracyCallback` sinkron dengan output | ✅ Terpenuhi |
| **Saran 2** | Resolusi citra asli tidak seragam | Terbukti dengan eksplorasi variasi ukuran resolusi citra asli | ✅ Terpenuhi |
| **Saran 3** | Memiliki 3 kelas atau lebih | Menggunakan 4 kelas: `butterfly`, `chicken`, `horse`, `spider` | ✅ Terpenuhi |
| **Saran 4** | Evaluasi Tambahan (Confusion Matrix & Classification Report) | Visualisasi Heatmap Confusion Matrix dan Classification Report lengkap pada Testing Set | ✅ Terpenuhi |
| **Saran 5** | Inferensi menggunakan salah satu model | Inferensi model TF-Lite pada data testing dengan metrik confidence dan latency | ✅ Terpenuhi |

---

## 📂 Struktur Direktori Submission

```
submission/
├── saved_model/
│   ├── fingerprint.pb
│   ├── saved_model.pb
│   └── variables/
│       ├── variables.data-00000-of-00001
│       └── variables.index
├── tflite/
│   ├── label.txt
│   └── model.tflite
├── tfjs_model/
│   ├── group1-shard1of1.bin
│   └── model.json
├── notebook.ipynb
├── notebook.py
├── README.md
└── requirements.txt
```

---

## 🚀 Cara Menjalankan Notebook

1. Buat environment Python:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows
   ```
2. Instal dependensi:
   ```bash
   pip install -r requirements.txt
   ```
3. Buka dan jalankan `notebook.ipynb` menggunakan Jupyter Lab, VS Code, atau Google Colab (klik *Restart Kernel and Run All Cells*).
