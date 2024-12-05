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

python getData.py

```

Run this command to training this model

```py

python train.py

```

Run this command to test or evaluate this model

```py

python test.py

```

## Results

<h1 align="center">
   <img src="https://github.com/user-attachments/assets/2c3fc534-635e-4e68-9736-1642f3e027a3">
</h1>

<h1 align="center">
   <img src="https://github.com/user-attachments/assets/cdba663b-03d3-4e15-84aa-7933845d0808">
</h1>
