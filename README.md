# Retail Intelligence — Integrasi Data Mining & Business Intelligence

Aplikasi web lokal untuk tugas UAS, menggunakan Python Flask sebagai backend, HTML/CSS/JavaScript untuk antarmuka, Chart.js untuk visualisasi, Pandas untuk pengolahan data, mlxtend untuk Apriori, scikit-learn untuk K-Means, openpyxl untuk Excel, dan ReportLab untuk PDF.

## Jalankan di Windows

1. Instal Python 3.10+.
2. Buka terminal pada folder proyek.
3. Buat virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\activate
   ```
4. Instal dependensi:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. Jalankan aplikasi:

   ```powershell
   python app.py
   ```
6. Buka `http://127.0.0.1:5000`.

## Alur masuk aplikasi (Cover → Dashboard)

Saat membuka `http://127.0.0.1:5000`, Anda tiba di **halaman cover**, bukan langsung dashboard. Cover memuat:

- **Judul & tagline**: "Retail Intelligence — Integrasi Data Mining & Business Intelligence" dengan label **Ujian Tengah Semester**.
- **Identitas mata kuliah**: Business Intelligence, Semester 7.
- **Identitas pengembang (inti)**: Bernard M. Deny Akoit, NPM 51230129, Program Studi Teknologi Informasi, beserta foto.
- **KPI sistem**: jumlah modul, algoritma data mining, ukuran dataset contoh, dan format ekspor.
- **Tujuan sistem**: lima sasaran analisis (profiling, preprocessing, Apriori, K-Means, rekomendasi).
- **Alur pengerjaan**: enam tahap dari Upload & EDA sampai Rekomendasi & Ekspor.
- **Input "Mulai Sistem"**: masukkan nama analis, lalu tekan *Masuk ke Dashboard*. Nama tersimpan di sesi lokal dan tampil di topbar dashboard.

Setelah masuk, route aktif adalah `/dashboard`. Menu Reset mengembalikan Anda ke `/dashboard` dengan workspace bersih; buka `/` kembali bila ingin melihat cover.

## Alur penggunaan

1. Buka **Dataset Management**.
2. Unggah `.xlsx`, `.xls`, atau `.csv`. Untuk Excel, sheet `Data_Raw` dipilih otomatis jika tersedia; selain itu sheet pertama digunakan.
3. Periksa preview, lalu tekan **Proses EDA & Cleaning**.
4. Buka **Data Profiling & EDA**, **Data Preprocessing**, dan **Data Validation & Audit** untuk memeriksa hasil.
5. Jalankan Apriori dan K-Means dari halaman masing-masing.
6. Tinjau Dashboard BI, rekomendasi, kemudian ekspor Excel atau PDF.
7. Gunakan Reset untuk menghapus workspace aktif sebelum mengunggah dataset lain.

## Fitur visualisasi per modul

| Modul | Visualisasi |
| --- | --- |
| EDA | Bar missing values per kolom, doughnut tipe data |
| Preprocessing | Grafik alur baris per tahap, boxplot kolom numerik, tabel statistik deskriptif (mean, median, std, kuartil, skewness, outlier IQR) |
| Validation & Audit | Bar perbandingan missing sebelum vs sesudah, bar temuan audit per tingkat keparahan, boxplot data mentah vs data bersih, tabel pemeriksaan aturan bisnis |
| Apriori | Panel kesiapan angka (transaksi, produk, baris basket, rata-rata item/basket, support disarankan), bar top itemset, scatter support vs confidence berwarna lift, tabel itemset + rules |
| K-Means | Elbow & silhouette, boxplot RFM per cluster, proyeksi 2D PCA, scatter interaktif Frequency vs Monetary, bar ukuran cluster + Monetary mean |
| Rekomendasi | Doughnut prioritas, bar top produk, bar rule lift terkuat, bar nilai cluster |

Angka yang mencapai ribuan ditampilkan dengan pemisah ribuan format Indonesia (titik), contoh `18.513` baris.

## Catatan penting

- Data sumber tidak dimodifikasi oleh preprocessing.
- Hanya baris identik yang dihapus sebagai duplikasi; ID transaksi saja tidak dipakai sebagai kunci duplikasi.
- Nilai numerik bermasalah ditandai untuk audit, bukan otomatis dihapus.
- Total penjualan pada dashboard merupakan agregasi awal dari kolom total yang terdeteksi. Verifikasi aturan bisnis dan diskon dari Data Dictionary sebelum pelaporan final.
- Halaman Apriori menampilkan **angka kesiapan** (jumlah transaksi, produk unik, baris basket valid, rata-rata item per basket) dan **nilai support/confidence yang disarankan** sebelum tombol dijalankan.
- Apriori memakai `mlxtend` bila tersedia, dan otomatis jatuh ke **engine cadangan berbasis Pandas** bila `mlxtend` tidak terinstal. Bila ambang yang dimasukkan tidak menghasilkan itemset/rule, sistem menurunkan ambang secara otomatis dan **mencatat penyesuaian tersebut** di panel hasil.
- K-Means memilih k dari silhouette tertinggi; hasil cluster divisualisasikan dengan boxplot (Matplotlib), proyeksi PCA 2D, dan scatter interaktif (Chart.js).
- Workspace disimpan dalam memori proses dan dirancang untuk demonstrasi lokal satu pengguna. Restart server menghapus state aktif. Ekspor yang sudah diunduh tetap aman.
- Untuk dataset dengan skema kolom berbeda, aplikasi mencoba memetakan nama kolom yang umum. Dataset yang tidak memiliki kolom inti akan memberi pesan error pada proses algoritma.
- Ekspor PDF saat ini merupakan laporan ringkasan metodologi dan hasil; tata letak dapat diperluas setelah hasil aktual ditinjau.

## Pengembang

**Bernard M. Deny Akoit** — NPM 51230129 — Program Studi Teknologi Informasi.
Profil tersedia pada menu **Pengembang** di sidebar.

## Pengujian

```powershell
$env:PYTHONIOENCODING='utf-8'
python tests/test_cover.py         # halaman cover & alur masuk (route / dan /dashboard)
python tests/smoke_test.py         # alur end-to-end: upload → analyze → apriori → kmeans → semua halaman → ekspor
python tests/verify_render.py      # verifikasi format ribuan, chart, dan halaman pengembang
python tests/test_apriori_kmeans.py # perilaku Apriori (ambang disarankan, relaksasi otomatis, fallback) & K-Means
```

## Status implementasi awal

Aplikasi menyediakan upload, profiling, preprocessing konservatif, audit aturan bisnis dengan tingkat keparahan, Apriori (dengan engine cadangan dan relaksasi ambang otomatis), RFM/K-Means (dengan evaluasi silhouette, boxplot, dan proyeksi cluster), ringkasan BI, rekomendasi berbasis hasil aktual, halaman profil pengembang, serta ekspor Excel/PDF. Seluruh modul analisis memiliki visualisasi grafik, tabel, atau boxplot — bukan sekadar angka. Pengujian penuh di setiap komputer perlu dilakukan setelah dependensi terinstal. Jangan menganggap semua analisis berhasil sebelum status proses aktual menunjukkan sukses.

## Struktur

```text
retail_intelligence/
├── app.py            # Flask app, preprocessing, Apriori, K-Means, audit, ekspor
├── charts.py         # Helper Matplotlib (boxplot, elbow, scatter cluster, dll.)
├── requirements.txt
├── README.md
├── deko.jpg          # Foto pengembang (sumber)
├── templates/
│   ├── base.html
│   ├── cover.html     # Halaman cover pembuka (judul, identitas, tujuan, alur, form mulai)
│   ├── index.html     # Dashboard inti (/dashboard)
│   └── page.html      # Semua halaman modul termasuk Pengembang
├── static/
│   ├── css/app.css
│   ├── img/deko.jpg  # Foto pengembang untuk halaman Pengembang
│   └── js/
│       ├── app.js
│       ├── dashboard.js   # Chart.js: KPI, cluster, rekomendasi
│       └── modules.js     # Form upload, Apriori, K-Means
├── tests/
│   ├── smoke_test.py
│   ├── verify_render.py
│   └── test_apriori_kmeans.py
├── uploads/
└── outputs/
```
