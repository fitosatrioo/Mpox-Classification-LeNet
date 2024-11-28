# <h1 align="center">Mpox Skin Lesion Classification With Deep Learning</h1>
<p align="center">Fito Satrio</p>
<p align="center">2311110030</p>

## Introduction
<p align="center">
<a href=""><img src="https://github.com/user-attachments/assets/b58c2d25-0999-47ce-9f92-bf25c6036e1a" alt="Build Status"></a>
<p/>

Selama fase awal puncak wabah Mpox, tantangan signifikan muncul akibat tidak adanya dataset yang andal dan tersedia secara publik untuk deteksi Mpox. Peningkatan kasus Mpox yang cepat, dengan potensi penyebarannya hingga Eropa dan Amerika seperti yang disoroti oleh Organisasi Kesehatan Dunia, serta kemungkinan munculnya kasus Mpox di negara-negara Asia, menekankan urgensi penerapan deteksi berbantuan komputer sebagai alat yang sangat penting. Dalam konteks ini, diagnosis Mpox secara cepat menjadi tantangan yang semakin sulit. Ketika kemungkinan wabah Mpox mengancam negara-negara dengan populasi padat seperti Bangladesh, keterbatasan sumber daya yang tersedia membuat diagnosis cepat tidak dapat dicapai. Oleh karena itu, kebutuhan mendesak akan metode deteksi berbantuan komputer menjadi jelas.

Untuk mengatasi kebutuhan mendesak ini, pengembangan metode berbantuan komputer memerlukan sejumlah besar data yang beragam, termasuk gambar lesi kulit Mpox dari individu dengan jenis kelamin, etnis, dan warna kulit yang berbeda. Namun, kelangkaan data yang tersedia menjadi hambatan besar dalam upaya ini. Menanggapi situasi kritis ini, kelompok penelitian kami mengambil inisiatif untuk mengembangkan salah satu dataset pertama (MSLD) yang dirancang khusus untuk Mpox, mencakup berbagai kelas, termasuk sampel non-Mpox.

Dari Juni 2022 hingga Mei 2023, Dataset Lesi Kulit Mpox (MSLD) telah mengalami dua iterasi, menghasilkan versi saat ini, MSLD v2.0. Versi sebelumnya mencakup dua kelas: "Mpox" dan "Lainnya" (non-Mpox), di mana kelas "Lainnya" terdiri dari gambar lesi kulit cacar air dan campak, yang dipilih karena kemiripannya dengan Mpox. Berdasarkan keterbatasan yang teridentifikasi pada rilis awal, kami mengembangkan versi yang lebih lengkap dan lebih komprehensif, yaitu MSLD v2.0. Dataset yang diperbarui ini mencakup lebih banyak kelas dan menyediakan set gambar yang lebih beragam, cocok untuk klasifikasi multi-kelas.
https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20/data

## About Dataset
Dataset ini dibagi ke dalam dua folder:

**Original Images:** Folder ini mencakup subfolder bernama "FOLDS" yang berisi lima fold (fold1-fold5) untuk validasi silang 5-fold dengan gambar asli. Setiap fold memiliki folder terpisah untuk set Test, Train, dan Validation.

**Augmented Images:** Augmented images terdiri dari 5 folds (lipatan) yang di dalamnya hanya terdapat folder Train untuk masing-masing kelas

Dataset ini terdiri dari gambar-gambar dari enam kelas yang berbeda, yaitu Mpox (284 images), Chickenpox (75 images), Measles (55 images), Cowpox (66 images), Hand-foot-mouth disease or HFMD (161 images), and Healthy (114 images). Dataset ini mencakup 755 gambar original skin lesion yang berasal dari 541 pasien yang berbeda, memastikan sampel yang representatif.

## Commands
Buat file getData.py yang berisi tentang pengolahan data untuk mengimpor, mengonversi, atau memproses data dari sumber tertentu, seperti file di dalam path folder, ke dalam format yang dapat digunakan untuk analisis lebih lanjut atau pelatihan model.

```py
import os
import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self, base_folder_aug, base_folder_orig):
        """
        :param base_folder_aug: Path folder untuk Augmented Images
        :param base_folder_orig: Path folder untuk Original Images
        """
        self.dataset_aug = []
        self.dataset_train = []
        self.dataset_test = []
        self.dataset_valid = []
        onehot = np.eye(6)  # One-hot encoding untuk 6 kelas
        
        # Load data dari Augmented Images (Train saja)
        for fold_num in range(1, 6):
            aug_folder = os.path.join(base_folder_aug, f"fold{fold_num}_AUG/Train/")
            for class_idx, class_name in enumerate(os.listdir(aug_folder)):
                class_folder = os.path.join(aug_folder, class_name)
                for img_name in os.listdir(class_folder):
                    img_path = os.path.join(class_folder, img_name)
                    image = cv.resize(cv.imread(img_path), (32, 32)) / 255
                    self.dataset_aug.append([image, onehot[class_idx]])
        
        # Load data dari Original Images (Train, Test, Valid)
        for fold_num in range(1, 6):
            fold_folder = os.path.join(base_folder_orig, f"fold{fold_num}/")
            
            # Load Train
            train_folder = os.path.join(fold_folder, "Train/")
            for class_idx, class_name in enumerate(os.listdir(train_folder)):
                class_folder = os.path.join(train_folder, class_name)
                for img_name in os.listdir(class_folder):
                    img_path = os.path.join(class_folder, img_name)
                    image = cv.resize(cv.imread(img_path), (32, 32)) / 255
                    self.dataset_train.append([image, onehot[class_idx]])

            # Load Test
            test_folder = os.path.join(fold_folder, "Test/")
            for class_idx, class_name in enumerate(os.listdir(test_folder)):
                class_folder = os.path.join(test_folder, class_name)
                for img_name in os.listdir(class_folder):
                    img_path = os.path.join(class_folder, img_name)
                    image = cv.resize(cv.imread(img_path), (32, 32)) / 255
                    self.dataset_test.append([image, onehot[class_idx]])

            # Load Valid
            valid_folder = os.path.join(fold_folder, "Valid/")
            for class_idx, class_name in enumerate(os.listdir(valid_folder)):
                class_folder = os.path.join(valid_folder, class_name)
                for img_name in os.listdir(class_folder):
                    img_path = os.path.join(class_folder, img_name)
                    image = cv.resize(cv.imread(img_path), (32, 32)) / 255
                    self.dataset_valid.append([image, onehot[class_idx]])
        
       
        print(f"Augmented Images (Train): {len(self.dataset_aug)}")
        print(f"Original Images (Train): {len(self.dataset_train)}")
        print(f"Original Images (Test): {len(self.dataset_test)}")
        print(f"Original Images (Valid): {len(self.dataset_valid)}")

    def __len__(self):
        """Mengembalikan jumlah data di Augmented Images (default)."""
        return len(self.dataset_aug)

    def __getitem__(self, idx):
        """
        :param idx: Index data
        :return: Tuple (image, label) dalam format tensor
        """
        features, label = self.dataset_aug[idx]
        return (torch.tensor(features, dtype=torch.float32).permute(2, 0, 1),  # Convert to CHW format
                torch.tensor(label, dtype=torch.float32))


if __name__ == "__main__":
    # Paths ke dataset
    aug_path = "././Dataset/Augmented Images/Augmented Images/FOLDS_AUG/"
    orig_path = "././Dataset/Original Images/Original Images/FOLDS/"
    
    # Inisialisasi Data
    data = Data(base_folder_aug=aug_path, base_folder_orig=orig_path)
    
```

Buat folder dengan Models yang berisi tentang file - file model yang akan digunakan sebagai training. saya menggunakan model LeNet yang bisa dipanggil melalui library torch.nn

```py

import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, num_classes=6):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(-1, 16 * 5 * 5)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

kemudian lakukan training pada data original images dan augmented images train yang ada pada file train.py

```py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from Models.LeNet import LeNet 
from Utils.getData import Data  

def main():
    BATCH_SIZE = 4
    EPOCH = 30
    LEARNING_RATE = 0.001
    NUM_CLASSES = 6 

    # Paths to dataset
    aug_path = "./Dataset/Augmented Images/Augmented Images/FOLDS_AUG/"
    orig_path = "./Dataset/Original Images/Original Images/FOLDS/"
    dataset = Data(base_folder_aug=aug_path, base_folder_orig=orig_path)

    train_data = dataset.dataset_train + dataset.dataset_aug
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    # LeNet model
    model = LeNet(num_classes=NUM_CLASSES)

    # loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()  # Suitable for multi-class classification
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    for epoch in range(EPOCH):
        loss_train = 0.0
        correct_train = 0
        total_train = 0
        model.train()

        for batch_idx, (src, trg) in enumerate(train_loader):
            src = src.permute(0, 3, 1, 2).float()  
            trg = torch.argmax(trg, dim=1)

            pred = model(src)
            loss = loss_fn(pred, trg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

            _, predicted = torch.max(pred, 1)
            total_train += trg.size(0)
            correct_train += (predicted == trg).sum().item()

        accuracy_train = 100 * correct_train / total_train
        print(f"Epoch [{epoch + 1}/{EPOCH}],
        Train Loss: {loss_train / len(train_loader):.4f},
        Accuracy: {accuracy_train:.2f}%")

        train_losses.append(loss_train / len(train_loader))
    torch.save(model.state_dict(), "trained_model4.pth")

    # Plot the training loss
    plt.plot(range(EPOCH), train_losses, color="#3399e6", label='Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("./training.png")
    # print("Model saved to lenet_trained_model.pth")

if __name__ == "__main__":
    main()
```

Kemudian lakukan validation dan testing pada original images di file testing.py

```py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Models.LeNet import LeNet
from Utils.getData import Data

def evaluate_model(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for src, trg in data_loader:
            src = src.permute(0, 3, 1, 2).float()
            trg = torch.argmax(trg, dim=1)
            
            outputs = model(src)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(trg.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Konversi label ke one-hot encoding untuk AUC
    all_labels_onehot = np.eye(6)[all_labels]
    auc = roc_auc_score(all_labels_onehot, all_probs, multi_class='ovr', average='weighted')

    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, precision, recall, f1, auc, cm

def plot_confusion_matrix(cm, class_names, save_path="confusion_matrix.png"):
    """
    Plot heatmap dari confusion matrix dan simpan sebagai gambar.
    :param cm: Confusion Matrix
    :param class_names: Daftar nama kelas
    :param save_path: Path untuk menyimpan heatmap
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
     yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.close()

def main():
    BATCH_SIZE = 4
    LEARNING_RATE = 0.001
    NUM_CLASSES = 6 

    aug_path = "./Dataset/Augmented Images/Augmented Images/FOLDS_AUG/"
    orig_path = "./Dataset/Original Images/Original Images/FOLDS/"


    dataset = Data(base_folder_aug=aug_path, base_folder_orig=orig_path)
    test_data = dataset.dataset_test
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # model LeNet
    model = LeNet(num_classes=NUM_CLASSES)

    model.load_state_dict(torch.load("trained_model4.pth"))
    model.eval()

    # Evaluasi model pada data test
    accuracy, precision, recall, f1, auc, cm = evaluate_model(model, test_loader)

    # hasil evaluasi
    print("Evaluasi pada data test:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Visualisasi Confusion Matrix sebagai Heatmap
    class_names = ["Chickenpox", "Cowpox", "Healthy", "HFMD", "Measles", "Monkeypox"] 
    plot_confusion_matrix(cm, class_names, save_path="./confusion_matrix.png")
    # print("Confusion Matrix heatmap saved as confusion_matrix.png")

if __name__ == "__main__":
    main()
```

## Results

### 1. Plot The Training Loss With 30 Epoch
Untuk training data pada dataset ini, menggunakan 30 epoch. Nilai loss awal sekitar 1.6, lalu secara konsisten menurun hingga mencapai nilai mendekati 0.2 pada akhir epoch ke-30.
Ini menunjukkan bahwa model sedang belajar dengan baik dari data pelatihan. Penurunan loss berarti model semakin baik memprediksi hasil dengan mengurangi kesalahan selama pelatihan. mungkin bisa lebih optimal ketika melakukan 50 - 100 epoch. Kurva loss juga terlihat halus dan tidak ada lonjakan besar. Hal ini mengindikasikan bahwa learning rate yang digunakan sudah sesuai dan model terhindar dari masalah seperti diverging atau overfitting dalam proses pelatihan.


<p align="center">
<a href=""><img src="https://github.com/user-attachments/assets/b4907adb-e6f6-4277-9831-6129bd480eea" alt="Training"></a>
<p/>


### 2. Evaluation Metrics With Confusion Matrix, Accuracy, Precision, Recall, F1 Score and ROC-AUC

a.) Confusion Matrix

Confusion matrix menunjukkan distribusi prediksi model untuk setiap kelas (Chickenpox, Cowpox, Healthy, HFMD, Measles, Monkeypox) dibandingkan dengan label sebenarnya:
Diagonal utama (angka besar pada latar biru gelap): Ini menunjukkan jumlah sampel yang diklasifikasikan dengan benar oleh model.
Chickenpox: 42 benar.
Cowpox: 35 benar.
Healthy: 49 benar.
HFMD: 82 benar.
Measles: 30 benar.
Monkeypox: 125 benar.
Angka non-diagonal: Semua 0, yang berarti tidak ada kesalahan klasifikasi (model sempurna pada matriks ini).

b.) Metrik Evaluasi

**Accuracy (0.9973):**
Akurasi menunjukkan proporsi prediksi yang benar terhadap total sampel. Nilai 0.9973 (99.73%) berarti model hampir sempurna dalam memprediksi data uji.

**Precision (0.9973):**
Precision mengukur seberapa akurat prediksi positif model untuk masing-masing kelas. Nilai 0.9973 menunjukkan bahwa hampir semua prediksi yang diberi label positif adalah benar.

**Recall (0.9973):**
Recall mengukur sejauh mana model berhasil menemukan semua sampel yang relevan (positif sebenarnya). Nilai tinggi ini menunjukkan bahwa model sangat sedikit kehilangan prediksi sampel yang benar.

**F1 Score (0.9972):**
F1 score adalah rata-rata harmonis dari precision dan recall. Nilai 0.9972 menunjukkan keseimbangan yang sangat baik antara precision dan recall.

**ROC-AUC (1.0000):**
ROC-AUC menunjukkan seberapa baik model dapat membedakan antara kelas-kelas yang berbeda. Nilai 1.0000 menunjukkan model memiliki kemampuan sempurna dalam membedakan kelas.

<p align="center">
<a href=""><img src="https://github.com/user-attachments/assets/c6ecea06-0cbc-4ec8-ac24-1c6e0ce49682" alt="Confusion Matrix"></a>
<p/>

<p align="center">
<a href=""><img src="https://github.com/user-attachments/assets/a5af7887-8d72-4764-863c-030aed248162" alt="Confusion Matrix"></a>
<p/>

## Citation
