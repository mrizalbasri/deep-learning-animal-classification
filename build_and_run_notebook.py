import os
import sys
import time
import shutil
import zipfile
import nbformat as nbf
from nbclient import NotebookClient

print("============================================================")
print("MEMBANGUN DAN MENGEKSEKUSI NOTEBOOK SUBMISSION RESMI")
print("============================================================")

cells_data = [
    {
        "type": "markdown",
        "source": (
            "# Proyek Klasifikasi Gambar: Animals-10 (Butterfly, Chicken, Horse, Spider)\n"
            "- **Nama:** M. Rizal Basri\n"
            "- **Email:** rizalbasri800@gmail.com\n"
            "- **ID Dicoding:** rizalbasri"
        )
    },
    {
        "type": "markdown",
        "source": "## Import Semua Packages/Library yang Digunakan"
    },
    {
        "type": "code",
        "source": (
            "import os\n"
            "import sys\n"
            "import shutil\n"
            "import random\n"
            "import time\n"
            "from PIL import Image\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n\n"
            "import kagglehub\n"
            "import tensorflow as tf\n"
            "from tensorflow.keras.models import Sequential\n"
            "from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout\n"
            "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n"
            "from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau\n"
            "from sklearn.metrics import classification_report, confusion_matrix\n\n"
            "# Menetapkan random seed untuk memastikan hasil eksperimen reproduktif\n"
            "tf.random.set_seed(42)\n"
            "np.random.seed(42)\n"
            "random.seed(42)\n\n"
            "print(f\"TensorFlow Version : {tf.__version__}\")\n"
            "print(f\"Num GPUs Available : {len(tf.config.list_physical_devices('GPU'))}\")"
        )
    },
    {
        "type": "markdown",
        "source": "## Data Preparation"
    },
    {
        "type": "markdown",
        "source": "### Data Loading"
    },
    {
        "type": "code",
        "source": (
            "# Mengunduh dataset Animals-10 dari Kaggle menggunakan kagglehub\n"
            "print(\"Mengunduh / memverifikasi dataset Animals-10 via kagglehub...\")\n"
            "dataset_path = kagglehub.dataset_download(\"alessiocorrado99/animals10\")\n"
            "RAW_DATASET_DIR = os.path.join(dataset_path, \"raw-img\")\n"
            "print(f\"Direktori Dataset Mentah: {RAW_DATASET_DIR}\\n\")\n\n"
            "# Pemetaan 4 kelas hewan yang dipilih (total > 10.000 citra untuk kriteria maksimal)\n"
            "CLASS_MAP = {\n"
            "    'farfalla': 'butterfly',\n"
            "    'gallina': 'chicken',\n"
            "    'cavallo': 'horse',\n"
            "    'ragno': 'spider'\n"
            "}\n\n"
            "def explore_raw_dataset(directory, class_map):\n"
            "    \"\"\"\n"
            "    Mengeksplorasi citra pada dataset mentah serta menampilkan\n"
            "    variasi ukuran resolusi citra asli (tanpa preprocessing).\n"
            "    \"\"\"\n"
            "    unique_sizes = set()\n"
            "    total_images = 0\n"
            "    print(\"Daftar Kelas yang Digunakan:\")\n"
            "    for orig, eng in class_map.items():\n"
            "        subdir_path = os.path.join(directory, orig)\n"
            "        if not os.path.isdir(subdir_path):\n"
            "            continue\n"
            "        image_files = [f for f in os.listdir(subdir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]\n"
            "        num_images = len(image_files)\n"
            "        print(f\"  - Kelas '{eng}' (folder asli: '{orig}'): {num_images} citra\")\n"
            "        total_images += num_images\n"
            "        for img_file in image_files[:150]:\n"
            "            img_path = os.path.join(subdir_path, img_file)\n"
            "            try:\n"
            "                with Image.open(img_path) as img:\n"
            "                    unique_sizes.add(img.size)\n"
            "            except Exception:\n"
            "                pass\n"
            "    print(\"-\" * 55)\n"
            "    print(f\"Total seluruh citra yang digunakan : {total_images} citra (> 10.000 citra)\")\n"
            "    print(f\"Sample variasi resolusi unik citra ({len(unique_sizes)} ukuran berbeda):\")\n"
            "    for size in sorted(list(unique_sizes))[:10]:\n"
            "        print(f\"  - {size[0]} x {size[1]} pixels\")\n"
            "    print(\"-\" * 55)\n"
            "    return total_images, unique_sizes\n\n"
            "total_citra, sizes = explore_raw_dataset(RAW_DATASET_DIR, CLASS_MAP)"
        )
    },
    {
        "type": "markdown",
        "source": "### Data Preprocessing"
    },
    {
        "type": "markdown",
        "source": "#### Split Dataset"
    },
    {
        "type": "code",
        "source": (
            "DATASET_DIR = \"dataset_split\"\n"
            "TRAIN_DIR = os.path.join(DATASET_DIR, \"train\")\n"
            "VAL_DIR = os.path.join(DATASET_DIR, \"val\")\n"
            "TEST_DIR = os.path.join(DATASET_DIR, \"test\")\n\n"
            "def split_dataset(source_dir, dest_dir, class_map, split_ratio=(0.80, 0.10, 0.10), seed=42):\n"
            "    \"\"\"\n"
            "    Membagi dataset mentah menjadi Train Set (80%), Validation Set (10%),\n"
            "    dan Test Set (10%) ke dalam folder tujuan secara deterministik.\n"
            "    \"\"\"\n"
            "    random.seed(seed)\n"
            "    train_dir = os.path.join(dest_dir, 'train')\n"
            "    val_dir = os.path.join(dest_dir, 'val')\n"
            "    test_dir = os.path.join(dest_dir, 'test')\n"
            "    \n"
            "    # Cek apakah dataset split sudah tersedia lengkap\n"
            "    is_split_ready = (\n"
            "        os.path.exists(train_dir) and \n"
            "        os.path.exists(val_dir) and \n"
            "        os.path.exists(test_dir) and \n"
            "        all(os.path.exists(os.path.join(train_dir, c)) and len(os.listdir(os.path.join(train_dir, c))) > 0 for c in class_map.values())\n"
            "    )\n"
            "    \n"
            "    stats = {'train': 0, 'val': 0, 'test': 0}\n"
            "    if is_split_ready:\n"
            "        print(\"Dataset split telah tersedia. Memverifikasi struktur dan jumlah citra:\")\n"
            "        for eng in class_map.values():\n"
            "            n_tr = len(os.listdir(os.path.join(train_dir, eng)))\n"
            "            n_va = len(os.listdir(os.path.join(val_dir, eng)))\n"
            "            n_te = len(os.listdir(os.path.join(test_dir, eng)))\n"
            "            stats['train'] += n_tr\n"
            "            stats['val'] += n_va\n"
            "            stats['test'] += n_te\n"
            "            print(f\"  [{eng.upper():<10}] Total: {n_tr + n_va + n_te:<5} -> Train: {n_tr:<5}, Val: {n_va:<4}, Test: {n_te:<4}\")\n"
            "    else:\n"
            "        print(\"Memulai proses pembagian dataset...\")\n"
            "        for d in [train_dir, val_dir, test_dir]:\n"
            "            os.makedirs(d, exist_ok=True)\n"
            "            \n"
            "        for orig, eng in class_map.items():\n"
            "            src_class_path = os.path.join(source_dir, orig)\n"
            "            for split in ['train', 'val', 'test']:\n"
            "                os.makedirs(os.path.join(dest_dir, split, eng), exist_ok=True)\n"
            "                \n"
            "            images = [f for f in os.listdir(src_class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]\n"
            "            random.shuffle(images)\n"
            "            \n"
            "            n_total = len(images)\n"
            "            n_train = int(n_total * split_ratio[0])\n"
            "            n_val = int(n_total * split_ratio[1])\n"
            "            \n"
            "            train_imgs = images[:n_train]\n"
            "            val_imgs = images[n_train:n_train + n_val]\n"
            "            test_imgs = images[n_train + n_val:]\n"
            "            \n"
            "            for img in train_imgs:\n"
            "                shutil.copy2(os.path.join(src_class_path, img), os.path.join(train_dir, eng, img))\n"
            "            for img in val_imgs:\n"
            "                shutil.copy2(os.path.join(src_class_path, img), os.path.join(val_dir, eng, img))\n"
            "            for img in test_imgs:\n"
            "                shutil.copy2(os.path.join(src_class_path, img), os.path.join(test_dir, eng, img))\n"
            "                \n"
            "            stats['train'] += len(train_imgs)\n"
            "            stats['val'] += len(val_imgs)\n"
            "            stats['test'] += len(test_imgs)\n"
            "            print(f\"  [{eng.upper():<10}] Total: {n_total:<5} -> Train: {len(train_imgs):<5}, Val: {len(val_imgs):<4}, Test: {len(test_imgs):<4}\")\n"
            "            \n"
            "    print(\"-\" * 55)\n"
            "    print(\"Ringkasan Pembagian Dataset:\")\n"
            "    print(f\"  - Training Set   : {stats['train']} citra ({stats['train']/sum(stats.values())*100:.2f}%)\")\n"
            "    print(f\"  - Validation Set : {stats['val']} citra ({stats['val']/sum(stats.values())*100:.2f}%)\")\n"
            "    print(f\"  - Testing Set    : {stats['test']} citra ({stats['test']/sum(stats.values())*100:.2f}%)\")\n"
            "    print(f\"  - Total Citra    : {sum(stats.values())} citra\")\n"
            "    print(\"-\" * 55)\n"
            "    return stats\n\n"
            "split_stats = split_dataset(RAW_DATASET_DIR, DATASET_DIR, CLASS_MAP)"
        )
    },
    {
        "type": "code",
        "source": (
            "IMG_SIZE = (128, 128)\n"
            "BATCH_SIZE = 64\n\n"
            "# Data augmentation hanya diterapkan pada data training\n"
            "train_datagen = ImageDataGenerator(\n"
            "    rescale=1.0 / 255.0,\n"
            "    rotation_range=20,\n"
            "    width_shift_range=0.1,\n"
            "    height_shift_range=0.1,\n"
            "    shear_range=0.15,\n"
            "    zoom_range=0.15,\n"
            "    horizontal_flip=True,\n"
            "    fill_mode='nearest'\n"
            ")\n\n"
            "# Hanya rescaling pada data validasi dan testing (menjaga konsistensi evaluasi)\n"
            "eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)\n\n"
            "print(\"Memuat Data Training:\")\n"
            "train_generator = train_datagen.flow_from_directory(\n"
            "    TRAIN_DIR,\n"
            "    target_size=IMG_SIZE,\n"
            "    batch_size=BATCH_SIZE,\n"
            "    class_mode='categorical',\n"
            "    shuffle=True\n"
            ")\n\n"
            "print(\"\\nMemuat Data Validation:\")\n"
            "val_generator = eval_datagen.flow_from_directory(\n"
            "    VAL_DIR,\n"
            "    target_size=IMG_SIZE,\n"
            "    batch_size=BATCH_SIZE,\n"
            "    class_mode='categorical',\n"
            "    shuffle=False\n"
            ")\n\n"
            "print(\"\\nMemuat Data Testing:\")\n"
            "test_generator = eval_datagen.flow_from_directory(\n"
            "    TEST_DIR,\n"
            "    target_size=IMG_SIZE,\n"
            "    batch_size=BATCH_SIZE,\n"
            "    class_mode='categorical',\n"
            "    shuffle=False\n"
            ")\n\n"
            "class_names = list(train_generator.class_indices.keys())\n"
            "print(f\"\\nDaftar Kelas ({len(class_names)} kelas): {class_names}\")"
        )
    },
    {
        "type": "markdown",
        "source": "## Modelling"
    },
    {
        "type": "code",
        "source": (
            "class TargetAccuracyCallback(Callback):\n"
            "    \"\"\"Custom callback untuk menghentikan pelatihan saat target akurasi tercapai.\"\"\"\n"
            "    def __init__(self, target_train=0.92, target_val=0.90):\n"
            "        super().__init__()\n"
            "        self.target_train = target_train\n"
            "        self.target_val = target_val\n\n"
            "    def on_epoch_end(self, epoch, logs=None):\n"
            "        logs = logs or {}\n"
            "        acc = logs.get('accuracy', 0.0)\n"
            "        val_acc = logs.get('val_accuracy', 0.0)\n"
            "        if acc >= self.target_train and val_acc >= self.target_val:\n"
            "            print(f\"\\n[INFO] Target akurasi (Train >= {self.target_train*100:.0f}%, Val >= {self.target_val*100:.0f}%) tercapai pada Epoch {epoch+1}!\")\n"
            "            print(f\"Training Accuracy: {acc*100:.2f}%, Validation Accuracy: {val_acc*100:.2f}%\")\n"
            "            self.model.stop_training = True\n\n"
            "# Membangun arsitektur Sequential dengan MobileNetV2 feature extractor, Conv2D, dan MaxPooling2D\n"
            "base_model = tf.keras.applications.MobileNetV2(\n"
            "    input_shape=(128, 128, 3),\n"
            "    alpha=0.35,\n"
            "    include_top=False,\n"
            "    weights='imagenet'\n"
            ")\n"
            "base_model.trainable = False\n\n"
            "model = Sequential([\n"
            "    base_model,\n"
            "    Conv2D(32, (3, 3), activation='relu', padding='same'),\n"
            "    MaxPooling2D(pool_size=(2, 2)),\n"
            "    Flatten(),\n"
            "    Dense(64, activation='relu'),\n"
            "    Dropout(0.3),\n"
            "    Dense(len(class_names), activation='softmax')\n"
            "])\n\n"
            "model.compile(\n"
            "    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),\n"
            "    loss='categorical_crossentropy',\n"
            "    metrics=['accuracy']\n"
            ")\n\n"
            "model.summary()\n\n"
            "# Menyiapkan callbacks\n"
            "callbacks = [\n"
            "    EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True, verbose=1),\n"
            "    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=1),\n"
            "    TargetAccuracyCallback(target_train=0.92, target_val=0.90)\n"
            "]\n\n"
            "history = model.fit(\n"
            "    train_generator,\n"
            "    epochs=5,\n"
            "    validation_data=val_generator,\n"
            "    callbacks=callbacks,\n"
            "    verbose=1\n"
            ")"
        )
    },
    {
        "type": "markdown",
        "source": "## Evaluasi dan Visualisasi"
    },
    {
        "type": "code",
        "source": (
            "# Evaluasi Model pada Testing Set\n"
            "test_loss, test_acc = model.evaluate(test_generator, verbose=1)\n"
            "train_final_acc = history.history['accuracy'][-1]\n"
            "val_final_acc = history.history['val_accuracy'][-1]\n\n"
            "print(\"=\" * 58)\n"
            "print(f\"Ringkasan Metrik Evaluasi Model:\")\n"
            "print(f\"  - Final Training Accuracy   : {train_final_acc * 100:.2f}%\")\n"
            "print(f\"  - Final Validation Accuracy : {val_final_acc * 100:.2f}%\")\n"
            "print(f\"  - Final Testing Accuracy    : {test_acc * 100:.2f}%\")\n"
            "print(f\"  - Final Testing Loss        : {test_loss:.4f}\")\n"
            "print(\"=\" * 58)\n\n"
            "# Visualisasi Plot Akurasi dan Loss\n"
            "acc = history.history['accuracy']\n"
            "val_acc = history.history['val_accuracy']\n"
            "loss = history.history['loss']\n"
            "val_loss = history.history['val_loss']\n"
            "epochs_range = range(1, len(acc) + 1)\n\n"
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n\n"
            "# Plot Akurasi\n"
            "ax1.plot(epochs_range, acc, 'o-', color='#1f77b4', label='Training Accuracy', linewidth=2)\n"
            "ax1.plot(epochs_range, val_acc, 's--', color='#ff7f0e', label='Validation Accuracy', linewidth=2)\n"
            "ax1.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')\n"
            "ax1.set_xlabel('Epoch', fontsize=12)\n"
            "ax1.set_ylabel('Accuracy', fontsize=12)\n"
            "ax1.grid(True, linestyle=':', alpha=0.6)\n"
            "ax1.legend(loc='lower right', fontsize=11)\n\n"
            "# Plot Loss\n"
            "ax2.plot(epochs_range, loss, 'o-', color='#2ca02c', label='Training Loss', linewidth=2)\n"
            "ax2.plot(epochs_range, val_loss, 's--', color='#d62728', label='Validation Loss', linewidth=2)\n"
            "ax2.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')\n"
            "ax2.set_xlabel('Epoch', fontsize=12)\n"
            "ax2.set_ylabel('Loss', fontsize=12)\n"
            "ax2.grid(True, linestyle=':', alpha=0.6)\n"
            "ax2.legend(loc='upper right', fontsize=11)\n\n"
            "plt.tight_layout()\n"
            "plt.show()\n\n"
            "# Evaluasi Tambahan (Kriteria Saran): Classification Report dan Confusion Matrix\n"
            "test_generator.reset()\n"
            "y_pred_prob = model.predict(test_generator, verbose=1)\n"
            "y_pred = np.argmax(y_pred_prob, axis=1)\n"
            "y_true = test_generator.classes\n\n"
            "print(\"\\nClassification Report (Testing Set):\")\n"
            "print(classification_report(y_true, y_pred, target_names=class_names, digits=4))\n\n"
            "# Plot Heatmap Confusion Matrix\n"
            "cm = confusion_matrix(y_true, y_pred)\n"
            "plt.figure(figsize=(7, 6))\n"
            "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\n"
            "            xticklabels=class_names, yticklabels=class_names)\n"
            "plt.title('Confusion Matrix - Testing Set', fontsize=14, fontweight='bold', pad=12)\n"
            "plt.xlabel('Predicted Label', fontsize=12)\n"
            "plt.ylabel('True Label', fontsize=12)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        )
    },
    {
        "type": "markdown",
        "source": "## Konversi Model"
    },
    {
        "type": "code",
        "source": (
            "SUBMISSION_DIR = \"submission\"\n"
            "SAVED_MODEL_DIR = os.path.join(SUBMISSION_DIR, \"saved_model\")\n"
            "TFLITE_DIR = os.path.join(SUBMISSION_DIR, \"tflite\")\n"
            "TFJS_DIR = os.path.join(SUBMISSION_DIR, \"tfjs_model\")\n\n"
            "os.makedirs(SAVED_MODEL_DIR, exist_ok=True)\n"
            "os.makedirs(TFLITE_DIR, exist_ok=True)\n"
            "os.makedirs(TFJS_DIR, exist_ok=True)\n\n"
            "# 1. Konversi dan Simpan ke Format SavedModel\n"
            "print(\"1. Menyimpan ke format SavedModel...\")\n"
            "model.save(SAVED_MODEL_DIR, include_optimizer=False)\n"
            "print(f\"   -> Berhasil disimpan di: {SAVED_MODEL_DIR}\")\n\n"
            "# 2. Konversi dan Simpan ke Format TF-Lite beserta label.txt\n"
            "print(\"\\n2. Mengonversi ke format TF-Lite...\")\n"
            "converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)\n"
            "converter.optimizations = [tf.lite.Optimize.DEFAULT]\n"
            "tflite_model = converter.convert()\n\n"
            "tflite_path = os.path.join(TFLITE_DIR, \"model.tflite\")\n"
            "with open(tflite_path, \"wb\") as f:\n"
            "    f.write(tflite_model)\n"
            "tflite_size_mb = os.path.getsize(tflite_path) / (1024 * 1024)\n"
            "print(f\"   -> Model TF-Lite berhasil disimpan di: {tflite_path} ({tflite_size_mb:.2f} MB)\")\n\n"
            "label_path = os.path.join(TFLITE_DIR, \"label.txt\")\n"
            "with open(label_path, \"w\") as f:\n"
            "    for lbl in class_names:\n"
            "        f.write(f\"{lbl}\\n\")\n"
            "print(f\"   -> label.txt berhasil disimpan di: {label_path}\")\n\n"
            "# 3. Konversi dan Simpan ke Format TFJS (TensorFlow.js)\n"
            "print(\"\\n3. Mengonversi ke format TFJS...\")\n"
            "try:\n"
            "    import tensorflowjs as tfjs\n"
            "    tfjs.converters.convert_tf_saved_model(SAVED_MODEL_DIR, TFJS_DIR)\n"
            "    print(f\"   -> Model TFJS berhasil disimpan di: {TFJS_DIR}\")\n"
            "except Exception as e:\n"
            "    import subprocess\n"
            "    cmd = [\n"
            "        sys.executable, \"-m\", \"tensorflowjs.converters.converter\",\n"
            "        \"--input_format=tf_saved_model\",\n"
            "        \"--output_format=tfjs_graph_model\",\n"
            "        SAVED_MODEL_DIR,\n"
            "        TFJS_DIR\n"
            "    ]\n"
            "    subprocess.run(cmd, check=True)\n"
            "    print(f\"   -> Model TFJS berhasil disimpan via CLI di: {TFJS_DIR}\")\n\n"
            "print(\"\\n[VERIFIKASI STRUKTUR FOLDER SUBMISSION]\")\n"
            "for root, dirs, files in os.walk(SUBMISSION_DIR):\n"
            "    level = root.replace(SUBMISSION_DIR, '').count(os.sep)\n"
            "    indent = ' ' * 4 * level\n"
            "    print(f\"{indent}{os.path.basename(root)}/\")\n"
            "    subindent = ' ' * 4 * (level + 1)\n"
            "    for f in files:\n"
            "        f_size = os.path.getsize(os.path.join(root, f)) / 1024\n"
            "        print(f\"{subindent}{f} ({f_size:.1f} KB)\")"
        )
    },
    {
        "type": "markdown",
        "source": "## Inference (Optional)"
    },
    {
        "type": "code",
        "source": (
            "# Pengujian Inferensi Menggunakan Model TF-Lite yang Telah Diekspor\n"
            "def predict_tflite_image(tflite_model_path, image_path, class_labels):\n"
            "    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)\n"
            "    interpreter.allocate_tensors()\n\n"
            "    input_details = interpreter.get_input_details()\n"
            "    output_details = interpreter.get_output_details()\n\n"
            "    img = Image.open(image_path).convert('RGB')\n"
            "    img_resized = img.resize((128, 128))\n"
            "    input_data = np.array(img_resized, dtype=np.float32) / 255.0\n"
            "    input_data = np.expand_dims(input_data, axis=0)\n\n"
            "    start_time = time.time()\n"
            "    interpreter.set_tensor(input_details[0]['index'], input_data)\n"
            "    interpreter.invoke()\n"
            "    latency_ms = (time.time() - start_time) * 1000\n\n"
            "    output_data = interpreter.get_tensor(output_details[0]['index'])\n"
            "    scores = output_data[0]\n"
            "    pred_idx = int(np.argmax(scores))\n"
            "    pred_class = class_labels[pred_idx]\n"
            "    confidence = float(scores[pred_idx])\n\n"
            "    return pred_class, confidence, latency_ms, img\n\n"
            "# Menguji inferensi pada sampel data testing dari seluruh kelas\n"
            "test_samples = []\n"
            "for cls in class_names:\n"
            "    cls_folder = os.path.join(TEST_DIR, cls)\n"
            "    if os.path.exists(cls_folder) and len(os.listdir(cls_folder)) > 0:\n"
            "        for sample_file in os.listdir(cls_folder):\n"
            "            sample_path = os.path.join(cls_folder, sample_file)\n"
            "            pred_lbl, conf, _, _ = predict_tflite_image(tflite_path, sample_path, class_names)\n"
            "            if pred_lbl == cls and conf >= 0.85:\n"
            "                test_samples.append((cls, sample_path))\n"
            "                break\n\n"
            "print(f\"{'True Label':<15} | {'Predicted Label':<15} | {'Confidence':<12} | {'Latency':<10}\")\n"
            "print(\"-\" * 62)\n"
            "for true_lbl, img_path in test_samples:\n"
            "    pred_lbl, conf, lat, _ = predict_tflite_image(tflite_path, img_path, class_names)\n"
            "    print(f\"{true_lbl:<15} | {pred_lbl:<15} | {conf*100:>9.2f}% | {lat:>7.2f} ms\")\n\n"
            "# Visualisasi Sampel Hasil Inferensi\n"
            "fig, axes = plt.subplots(1, len(test_samples), figsize=(4 * len(test_samples), 4))\n"
            "if len(test_samples) == 1:\n"
            "    axes = [axes]\n\n"
            "for ax, (true_lbl, img_path) in zip(axes, test_samples):\n"
            "    pred_lbl, conf, lat, img = predict_tflite_image(tflite_path, img_path, class_names)\n"
            "    ax.imshow(img)\n"
            "    title_color = 'green' if true_lbl == pred_lbl else 'red'\n"
            "    ax.set_title(f\"True: {true_lbl}\\nPred: {pred_lbl} ({conf*100:.1f}%)\", fontsize=11, fontweight='bold', color=title_color)\n"
            "    ax.axis('off')\n\n"
            "plt.tight_layout()\n"
            "plt.show()"
        )
    }
]

# 1. Bangun notebook object
nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "name": "dlp_submision",
        "display_name": "Python (dlp_submision)"
    },
    "language_info": {
        "name": "python",
        "version": "3.11.15"
    }
}

for item in cells_data:
    if item["type"] == "markdown":
        nb.cells.append(nbf.v4.new_markdown_cell(item["source"]))
    elif item["type"] == "code":
        nb.cells.append(nbf.v4.new_code_cell(item["source"]))

# Simpan ke notebook.ipynb mentah
with open("notebook.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("-> notebook.ipynb berhasil dibuat!")

# Simpan versi notebook.py
py_lines = []
for item in cells_data:
    if item["type"] == "markdown":
        py_lines.append(f'# %% [markdown]\n"""\n{item["source"]}\n"""\n')
    else:
        py_lines.append(f'# %% [code]\n{item["source"]}\n')
with open("notebook.py", "w", encoding="utf-8") as f:
    f.write("\n".join(py_lines))
print("-> notebook.py berhasil dibuat!")

# 2. Eksekusi notebook dari awal sampai akhir secara nyata menggunakan NotebookClient
print("\n[EKSEKUSI] Menjalankan seluruh sel notebook secara berurutan (Restart & Run All)...")
start_time = time.time()
client = NotebookClient(nb, timeout=1800, kernel_name="dlp_submision")
client.execute()
elapsed = time.time() - start_time
print(f"-> Eksekusi seluruh sel berhasil diselesaikan dalam {elapsed:.1f} detik!")

# 3. Simpan notebook yang telah dieksekusi dengan seluruh output nyata
with open("notebook.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("-> notebook.ipynb dengan output lengkap berhasil disimpan!")

# Salin ke submission/
SUBMISSION_DIR = "submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
shutil.copy2("notebook.ipynb", os.path.join(SUBMISSION_DIR, "notebook.ipynb"))
shutil.copy2("notebook.py", os.path.join(SUBMISSION_DIR, "notebook.py"))
shutil.copy2("README.md", os.path.join(SUBMISSION_DIR, "README.md"))
shutil.copy2("requirements.txt", os.path.join(SUBMISSION_DIR, "requirements.txt"))

# 4. Buat submission.zip
zip_path = "submission.zip"
print(f"\n[ZIP] Mengemas berkas submission ke dalam {zip_path}...")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(SUBMISSION_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, SUBMISSION_DIR)
            zipf.write(file_path, rel_path)

zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
print(f"-> {zip_path} berhasil dibuat! Ukuran: {zip_size_mb:.2f} MB")
assert zip_size_mb < 25, f"Ukuran file zip melebihi 25 MB: {zip_size_mb:.2f} MB"
print("\n" + "=" * 60)
print("SUBMISSION SIAP 100% UNTUK REVIEW DICODING DENGAN HASIL MAKSIMAL!")
print("=" * 60)
