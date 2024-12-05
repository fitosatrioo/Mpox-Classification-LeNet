# <h1 align="center">Mpox Skin Lesion Classification With Deep Learning</h1>
<p align="center">Fito Satrio</p>
<p align="center">2311110030</p>

## Introduction
<h1 align="center">
   <img src="https://github.com/user-attachments/assets/261feefd-60c4-4280-ac9f-c07f1fb49025">
</h1>

Selama fase awal puncak wabah Mpox, tantangan signifikan muncul akibat tidak adanya dataset yang andal dan tersedia secara publik untuk deteksi Mpox. Peningkatan kasus Mpox yang cepat, dengan potensi penyebarannya hingga Eropa dan Amerika seperti yang disoroti oleh Organisasi Kesehatan Dunia, serta kemungkinan munculnya kasus Mpox di negara-negara Asia, menekankan urgensi penerapan deteksi berbantuan komputer sebagai alat yang sangat penting. Dalam konteks ini, diagnosis Mpox secara cepat menjadi tantangan yang semakin sulit. Ketika kemungkinan wabah Mpox mengancam negara-negara dengan populasi padat seperti Bangladesh, keterbatasan sumber daya yang tersedia membuat diagnosis cepat tidak dapat dicapai. Oleh karena itu, kebutuhan mendesak akan metode deteksi berbantuan komputer menjadi jelas.

Untuk mengatasi kebutuhan mendesak ini, pengembangan metode berbantuan komputer memerlukan sejumlah besar data yang beragam, termasuk gambar lesi kulit Mpox dari individu dengan jenis kelamin, etnis, dan warna kulit yang berbeda. Namun, kelangkaan data yang tersedia menjadi hambatan besar dalam upaya ini. Menanggapi situasi kritis ini, kelompok penelitian kami mengambil inisiatif untuk mengembangkan salah satu dataset pertama (MSLD) yang dirancang khusus untuk Mpox, mencakup berbagai kelas, termasuk sampel non-Mpox.

Dari Juni 2022 hingga Mei 2023, Dataset Lesi Kulit Mpox (MSLD) telah mengalami dua iterasi, menghasilkan versi saat ini, MSLD v2.0. Versi sebelumnya mencakup dua kelas: "Mpox" dan "Lainnya" (non-Mpox), di mana kelas "Lainnya" terdiri dari gambar lesi kulit cacar air dan campak, yang dipilih karena kemiripannya dengan Mpox. Berdasarkan keterbatasan yang teridentifikasi pada rilis awal, kami mengembangkan versi yang lebih lengkap dan lebih komprehensif, yaitu MSLD v2.0. Dataset yang diperbarui ini mencakup lebih banyak kelas dan menyediakan set gambar yang lebih beragam, cocok untuk klasifikasi multi-kelas.

## About Dataset

Dataset ini diorganisir ke dalam dua folder:

**Original Images:** Folder ini mencakup subfolder bernama "FOLDS" yang berisi lima fold (fold1-fold5) untuk validasi silang 5-fold dengan gambar asli. Setiap fold memiliki folder terpisah untuk set uji, pelatihan, dan validasi.

**Augmented Images:** Untuk meningkatkan tugas klasifikasi, berbagai teknik augmentasi data. Agar hasilnya dapat direproduksi, gambar-gambar augmentasi disediakan dalam folder ini. Folder ini berisi subfolder bernama "FOLDS_AUG" yang berisi gambar-gambar augmentasi dari set pelatihan dari setiap fold dalam subfolder "FOLDS" pada folder "original images". Proses augmentasi menghasilkan peningkatan jumlah gambar sekitar 14 kali lipat.

Dataset ini terdiri dari gambar-gambar dari enam kelas yang berbeda, yaitu Mpox(284 images), Chickenpox (75 images), Measles (55 images), Cowpox (66 images), Hand-foot-mouth disease or HFMD (161 images), and Healthy (114 images). Dataset ini mencakup 755 original skin lesion images yang berasal dari 541 pasien yang berbeda, memastikan sampel yang representatif.

## Commands

Run this command to check whether the dataset has been called

```py

python getData.py

```

Creted model with LeNet.py
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
        x = x.reshape(-1, 16 * 5 * 5) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

Run this command to training this model

```py

python train.py

```

Run this command to test or evaluate this model

```py

python testing.py

```
**Output:**
<h1>
   <img src="https://github.com/user-attachments/assets/f589647d-6407-46ad-93ed-90b4279e9c09">
</h1>

## Results

### 1. Training With 50 Epoch

Dapat dilihat pada grafik dibawah ini, Training Loss menurun secara konsisten seiring bertambahnya epoch, yang menunjukkan bahwa model semakin mampu mempelajari pola dalam data pelatihan.
Penurunan yang stabil ini menunjukkan bahwa optimisasi model bekerja dengan baik. sedangkan pada Validation Loss juga menurun secara signifikan, terutama pada awal epoch. Validation Loss lebih rendah dibandingkan Training Loss setelah beberapa epoch, yang menunjukkan bahwa model generalisasi dengan baik pada data validasi. Dengan Validation Loss yang sangat rendah (hampir mendekati nol), dapat diinterpretasikan bahwa model memiliki performa yang sangat baik pada data validasi. Model LeNet pretrained ini menunjukkan performa yang baik dengan tidak adanya tanda overfitting dan loss yang konsisten menurun.

<h1 align="center">
   <img src="https://github.com/user-attachments/assets/2c3fc534-635e-4e68-9736-1642f3e027a3">
</h1>

### 2. Confusion Matrix and Metrics Evaluation  (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
### A.) Confusion Matrix Analysis
<h4>True Positive</h4>

Angka pada diagonal utama menunjukkan jumlah sampel yang diklasifikasikan dengan benar ke kelas aslinya.
- Chickenpox: 42 sampel diklasifikasikan benar sebagai Chickenpox.
- Cowpox: 35 sampel diklasifikasikan benar sebagai Cowpox.
- Healthy: 49 sampel diklasifikasikan benar sebagai Healthy.
- HFMD: 82 sampel diklasifikasikan benar sebagai HFMD.
- Measles: 30 sampel diklasifikasikan benar sebagai Measles.
- Monkeypox: 125 sampel diklasifikasikan benar sebagai Monkeypox.

<h4>Missclasification</h4>

Ada 1 sampel dari kelas Healthy salah diklasifikasikan sebagai Monkeypox. Ini terlihat dari angka 1 di baris "Healthy" (True Label) dan kolom "Monkeypox" (Predicted Label).


### B.) Metrics Evaluation

- **Accuracy: 0.9973 (99.73%)**
Ini Menunjukkan proporsi total prediksi yang benar dari semua kelas.

- **Precision: 0.9973 (99.73%)**
Menunjukkan ketepatan model dalam mengklasifikasikan kelas positif (berapa banyak prediksi benar dari total prediksi kelas positif).

- **Recall: 0.9973 (99.73%)**
Mengukur sensitivitas model, yaitu seberapa baik model mampu mendeteksi sampel dari kelas tertentu.

- **F1 Score: 0.9972 (99.72%)**
Kombinasi dari precision dan recall, yang menyeimbangkan keduanya. Nilai tinggi menunjukkan model yang sangat baik.

- **ROC-AUC: 1.000 (100%)**
Area under the ROC curve adalah 1.000, yang menunjukkan model memiliki performa sempurna dalam membedakan antar kelas.

Model LeNet pretrained menunjukkan kinerja yang sangat baik dalam tugas klasifikasi multi-kelas dengan akurasi sebesar 99.73%, yang mencerminkan proporsi prediksi yang benar dari seluruh data. Precision dan recall masing-masing sebesar 99.73% menunjukkan bahwa model ini memiliki ketepatan tinggi dalam mengklasifikasikan sampel positif serta sensitivitas yang baik dalam mendeteksi semua sampel dari setiap kelas. Kombinasi precision dan recall yang seimbang menghasilkan F1-score sebesar 99.72%, menandakan kemampuan model untuk menghindari bias terhadap salah satu metrik. Selain itu, nilai ROC-AUC sebesar 1.000 menunjukkan bahwa model ini mampu membedakan antar kelas dengan sempurna hanya dengan 50 epoch saja. **I dont know whether this model can do 100% prediction or not, i think it can -fito**


<h1 align="center">
   <img src="https://github.com/user-attachments/assets/cdba663b-03d3-4e15-84aa-7933845d0808">
</h1>


## Citation

