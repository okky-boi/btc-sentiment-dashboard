# Dashboard Bitcoin Big Data & TFT Forecast (Versi Online)

Dashboard Streamlit untuk studi kasus *Analisis Pergerakan Market Bitcoin Menggunakan Analisis
Sentimen Media Sosial Reddit* (mata kuliah Big Data Analytics).

Repository ini hanya berisi dashboard dan **snapshot lapisan Serving** dari data lake lokal. Pipeline
lengkap (HDFS, Apache Spark, training TFT) berjalan di laptop melalui Docker Compose dan tidak
disertakan di sini.

## Isi

| Path | Keterangan |
|---|---|
| `src/app.py` | Aplikasi Streamlit (Tab 1: harga, prediksi TFT, sentimen; Tab 2: graph analytics) |
| `src/config.py` | Konfigurasi path dan nama tabel Serving |
| `src/hdfs_client.py` | Pembaca tabel; jika HDFS tidak tersedia, membaca salinan lokal di `lake/serving/` |
| `lake/serving/*.parquet` | Snapshot tabel `price_actual`, `sentiment_hourly`, `price_forecast`, `graph_nodes`, `graph_edges` |

## Perbedaan dengan versi lokal

- Data tidak diperbarui otomatis. Dashboard online menampilkan snapshot terakhir yang di-push ke GitHub.
- HDFS tidak dapat dijangkau dari cloud, sehingga sidebar menampilkan status `LOCAL LAKEHOUSE MIRROR`.
- Data yang diunggah hanya agregat per jam; tidak ada teks post maupun identitas penulis Reddit.

## Menjalankan secara lokal

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

## Deploy ke Streamlit Community Cloud

1. Push isi folder ini ke repository GitHub (publik).
2. Buka https://share.streamlit.io, pilih **Create app** → **Deploy a public app from GitHub**.
3. Isi repository, branch `main`, dan **Main file path** `src/app.py`.
4. Di **Advanced settings**, pilih Python **3.12**, lalu **Deploy**.

## Memperbarui snapshot data

Salin ulang file dari `lake/serving/` di proyek lokal ke folder `lake/serving/` di repository ini, lalu
commit dan push. Streamlit Cloud akan memuat ulang aplikasi secara otomatis.
