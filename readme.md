# API untuk Membaca Model OCR Menggunakan Flask

## Pendahuluan
Proyek ini bertujuan untuk membuat API yang dapat membaca model OCR (Optical Character Recognition) menggunakan Flask, kemudian dikembangkan lebih lanjut dengan FastAPI. Proyek ini mencakup langkah-langkah dari preprocessing gambar hingga decoding hasil prediksi dari model.

## Fitur
- **Upload Gambar**: Mengunggah gambar untuk diproses oleh model OCR.
- **Prediksi Teks**: Menghasilkan prediksi dari gambar yang diunggah.
- **Multi-label Output**: Memetakan hasil prediksi ke label seperti `company`, `address`, `date`, dan `total`.
- **Logging**: Mendukung logging untuk debugging dan pelacakan kesalahan.

## Struktur Proyek
```
api-model-ocr/
|-- app.py                  # File utama untuk menjalankan API
|-- log_config.py           # Konfigurasi logging
|-- ocr_model.h5            # Model OCR yang digunakan
|-- train_dataset.csv       # Dataset pelatihan (opsional untuk debugging)
|-- example_image.jpg       # Contoh gambar untuk pengujian
```

## Instalasi

### 1. Kloning Repository
```bash
git clone <repository-url>
cd api-model-ocr
```

### 2. Instal Dependensi
Pastikan Anda telah menginstal Python 3.9+ dan pip. Lalu jalankan:
```bash
pip install -r requirements.txt
```

### 3. Jalankan API
Untuk Flask:
```bash
python app.py
```

## Cara Menggunakan Endpoint Utama

### Endpoint: **POST /predict**
- **Input**: File gambar (format JPEG/PNG).
- **Output**: JSON berisi prediksi teks.

#### Contoh Input:
Kirim gambar menggunakan alat seperti Postman:
```
POST http://127.0.0.1:5000/predict
Content-Type: multipart/form-data
File: example_image.jpg
```

#### Contoh Output:
```json
{
  "company": "CompanyName",
  "address": "AddressString",
  "date": "2024-12-13",
  "total": "$123.45"
}
```

## Tantangan dan Solusi

### Masalah Dimensi Input
- **Masalah**: Model membutuhkan input (128, 128, 1), tetapi preprocessing awal menghasilkan dimensi (32, 128, 1).
- **Solusi**: Resize ulang gambar menjadi (128, 128).

### Decoding Error
- **Masalah**: Error "string index out of range" saat mapping hasil prediksi.
- **Solusi**: Tambahkan validasi panjang teks sebelum mapping.

### Prediksi Tidak Sesuai
- **Masalah**: Hasil prediksi tidak sesuai dengan label yang diharapkan.
- **Solusi**: Periksa ulang preprocessing gambar dan arsitektur model OCR.

## Perbaikan yang Dapat Dilakukan
- **Pelatihan Ulang Model**: Gunakan dataset yang lebih terstruktur dengan label `company`, `address`, `date`, dan `total`.
- **Validasi Input**: Tambahkan validasi format file gambar.
- **Peningkatan Decoding**: Gunakan algoritma decoding yang lebih kompleks untuk menghasilkan prediksi yang lebih akurat.

## Dependensi
- Python 3.9+
- Flask
- TensorFlow
- Pillow
- NumPy

## Catatan Tambahan
- File `ocr_model.h5` adalah model pra-latih. Pastikan file ini kompatibel dengan TensorFlow yang Anda gunakan.
- Untuk memastikan prediksi berjalan dengan baik, pastikan input gambar sesuai dengan format yang diharapkan oleh model.

## Pengembang
Dikembangkan dengan fokus pada pembelajaran dan eksperimen menggunakan Flask dan FastAPI.
