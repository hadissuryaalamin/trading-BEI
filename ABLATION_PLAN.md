# ABLATION_PLAN — mencari jumlah fitur optimal

Rencana kerja untuk **feature ablation study**: menentukan berapa banyak (dan
kelompok fitur mana) yang menghasilkan strategi terbaik untuk `trading-BEI`.
Melengkapi [PLAN.md](PLAN.md); tidak mengubah keputusan inti (DLSA tanpa factor
model, Transformer end-to-end, walk-forward).

Konteks strategi: versi **long-only** (posisi di `[0, 1]`, bukan dollar-neutral
`[-1, 1]`). Konsekuensi: kena beta pasar, jadi **benchmark wajib IHSG
buy-and-hold**, bukan sekadar Sharpe absolut.

---

## 1. Tujuan & pertanyaan riset

**Pertanyaan utama:** berapa jumlah / kombinasi grup fitur yang memberi
performa out-of-sample terbaik, sebelum tambahan fitur mulai menurunkan hasil
(overfit)?

**Hipotesis:** karena tidak ada factor model, Transformer menanggung beban
belajar co-movement sendiri. Menambah fitur menaikkan kapasitas tapi juga risiko
overfit pada data terbatas (~1.250 hari bursa). Diperkirakan ada titik jenuh di
kisaran 10-15 fitur.

**Kriteria "optimal":** Sharpe walk-forward tertinggi yang tetap mengalahkan
benchmark IHSG, dengan drawdown & turnover wajar, dan stabil lintas seed.

---

## 2. Grup fitur

Ablation dilakukan **per grup ekonomi**, bukan per fitur satuan (menghindari
ledakan eksperimen dan noise). Lima grup:

| Grup | Nama | Fitur | Dimensi ekonomi |
|------|------|-------|-----------------|
| G1 | return & momentum | log_return, mom_5, mom_10, mom_20 | arah & tren harga |
| G2 | volatilitas/range | hl_range, roll_vol_20, range_ratio | risiko/volatilitas |
| G3 | volume & likuiditas | log_volume, log_value, turnover, amihud | aktivitas & likuiditas |
| G4 | aliran asing | foreign_flow_ratio, foreign_roll_5 | tekanan asing |
| G5 | mikrostruktur | bid_ask_spread, book_imbalance | order book (bid/offer) |

Catatan: setiap fitur dinormalisasi **cross-sectional per hari** (gaya DLSA) di
`normalize()`. Beberapa fitur juga bisa punya varian rolling z-score (vs sejarah
saham sendiri) — dipertimbangkan di fase lanjut, bukan ablation dasar.

---

## 3. Desain eksperimen (nested / bertingkat)

Mulai minimal, tambah satu grup tiap tahap. Titik saat Sharpe berhenti naik =
jumlah fitur optimal.

| Exp | Grup aktif | ~jml fitur |
|-----|-----------|-----------|
| A (baseline) | G1 | 4 |
| B | G1+G2 | 7 |
| C | G1+G2+G3 | 11 |
| D | G1+G2+G3+G4 | 13 |
| E (full) | G1+G2+G3+G4+G5 | 15 |

Opsional lanjutan setelah tahu grup mana yang paling berkontribusi:
- **leave-one-group-out** dari konfigurasi terbaik (konfirmasi tiap grup
  benar-benar menambah nilai).
- ablation halus di dalam grup pemenang.

---

## 4. Kontrol eksperimen (WAJIB)

Agar perbandingan valid, **hanya daftar fitur yang boleh berbeda** antar
eksperimen. Semua ini dikunci identik:

- arsitektur Transformer (d_model, layer, head, dropout)
- lookback `T`, horizon target
- split walk-forward (tanggal train/val/test sama persis)
- biaya transaksi (bps), penalti turnover, constraint long-only
- jumlah epoch, learning rate, early-stopping rule
- daftar seed

Tanpa kontrol ini, perbedaan Sharpe tidak bisa diatribusikan ke fitur.

---

## 5. Metrik & evaluasi

Evaluasi **out-of-sample (walk-forward)**, bukan loss training.

Metrik utama:
- **Sharpe** (annualized) — metrik peringkat utama
- **return kumulatif vs IHSG buy-and-hold** — karena long-only kena beta
- **max drawdown**
- **turnover** (proxy biaya nyata; long-only IDX biayanya tinggi)

Robustness:
- tiap eksperimen dijalankan **8 seed**; laporkan **mean ± std**.
- perbedaan Sharpe < ~0,1 kemungkinan noise seed, bukan efek fitur.
- 8 seed cukup untuk estimasi std yang stabil dan uji signifikansi antar-eksperimen.

---

## 6. Perubahan kode yang diperlukan

1. **`src/preprocess.py`** — jadikan config-driven.
   - ganti konstanta `FEATURE_COLUMNS` dengan definisi **grup fitur**.
   - `compute_features()` menghitung superset semua fitur; grup yang dipakai
     dipilih dari config.
   - semua fitur tetap kausal (tanpa look-ahead).
2. **Bugfix `open=0`** — data IDX sering `OpenPrice=0`. Perbaiki
   `overnight_return`/fitur berbasis open: fallback `open→prev_close` atau
   tandai invalid, agar tidak memicu `dropna` massal.
3. **`configs/`** — satu YAML per eksperimen (A-E), hanya beda field
   `feature_groups`, sisanya identik (idealnya via config dasar bersama).
4. **`compare.py`** — kumpulkan hasil semua eksperimen jadi satu tabel:
   fitur/grup → Sharpe → return → drawdown → turnover (mean ± std antar-seed).
5. **`results/`** — simpan metrik + plot equity curve per eksperimen vs IHSG.

---

## 7. Alur kerja

1. Regenerate `data/processed/panel.parquet` (file sekarang korup — footer
   parquet tidak terbaca).
2. Refactor `preprocess.py` jadi grup fitur + bugfix open.
3. Buat config dasar + 5 config eksperimen (A-E).
4. Latih tiap eksperimen × 8 seed (walk-forward).
5. Kumpulkan metrik via `compare.py`.
6. Analisis: di grup mana Sharpe jenuh/menurun → tentukan set fitur optimal.
7. (Opsional) leave-one-group-out untuk konfirmasi.
8. Dokumentasikan temuan di `results/` + ringkasan di sini.

---

## 8. Risiko & catatan

- **Overfit** akibat data terbatas + banyak fitur → itu justru yang diuji;
  pertahankan regularisasi & early-stopping konsisten.
- **Kolinearitas** (mis. log_volume vs log_value) → jangan tafsirkan fitur
  redundan sebagai "informasi baru".
- **Noise seed** → wajib multi-seed, jangan simpulkan dari satu run.
- **Look-ahead** pada fitur momentum/rolling & normalisasi → titik paling rawan
  bug; uji dengan unit test anti look-ahead.
- **Benchmark** long-only harus IHSG, bukan return absolut.

---

## 9. Definition of done

- 5 eksperimen (A-E) selesai, masing-masing 8 seed (40 run total).
- Tabel pembanding terisi (Sharpe/return/drawdown/turnover, mean ± std).
- Set fitur optimal teridentifikasi dengan justifikasi (titik jenuh Sharpe).
- Temuan terdokumentasi di `results/` dan ringkas di file ini.
