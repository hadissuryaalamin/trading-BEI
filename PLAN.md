# PLAN — trading-BEI

Deep-learning statistical arbitrage untuk saham IDX, adaptasi dari
[DLSA (dlsa-public)](https://github.com/gregzanotti/dlsa-public).

**Perbedaan inti dari DLSA asli:** kita **tidak pakai factor model**. Di DLSA,
raw data → factor model → residual → trading policy. Di sini raw data IDX
langsung dinormalisasi lalu **dimasukkan ke Transformer** yang belajar
representasi + sinyal trading secara end-to-end. Tahap `factor_models/` dan
`residuals/` dihapus.

Keputusan yang sudah disepakati:
- **Sumber data:** situs resmi IDX (endpoint *Ringkasan Saham / Stock Summary*).
- **Universe:** semua saham BEI (~900 emiten), lalu difilter kualitas.
- **Framework:** PyTorch.
- **Periode:** ~5 tahun harian (mis. 2020-07-01 → 2025-06-30).

---

## Bagian 1 — Scraping stock summary IDX (harian, 5 tahun)

**Tujuan:** satu file per hari bursa berisi ringkasan semua saham.

**Sumber.** IDX menyediakan JSON per tanggal lewat endpoint trading summary:

```
https://www.idx.co.id/primary/TradingSummary/GetStockSummary?length=9999&start=0&date=YYYYMMDD
```

Setiap baris = satu saham, dengan field antara lain: `Open/High/Low/Close`,
`Previous`, `Volume`, `Value`, `Frequency`, `ForeignBuy`, `ForeignSell`,
`ListedShares`, `Bid/Offer`, dll. (nama field harus dikonfirmasi ulang karena
struktur situs bisa berubah).

**Langkah kerja** (`scraper/idx_scraper.py`):

1. Iterasi rentang tanggal, lewati Sabtu/Minggu; hari libur bursa otomatis
   ketahuan karena respons kosong.
2. GET per tanggal dengan retry + exponential backoff (`tenacity`), plus jeda
   antar-request (default 1 dtk) supaya sopan terhadap server.
3. Simpan mentah per hari ke `data/raw/idx_YYYYMMDD.json` (atau parquet).
4. Flag `--resume` untuk skip tanggal yang sudah terunduh (aman diulang).

**Risiko & mitigasi:**
- Endpoint mungkin butuh header/cookie tertentu → tangani session/headers.
- Struktur/field bisa berubah → validasi skema saat `build_panel`.
- Volume request besar (~1.250 hari bursa) → rate-limit + resume wajib.
- Cadangan bila endpoint diblok: fallback `yfinance` (ticker `.JK`) untuk OHLCV.

**Output tahap ini:** `data/raw/` terisi ~1.250 file harian.

---

## Bagian 2 — Skeleton folder data & arsitektur model

### Struktur folder (sudah di-scaffold)

```
trading-BEI/
├── README.md              overview
├── PLAN.md                dokumen ini
├── requirements.txt
├── LICENSE
├── .gitignore
├── configs/               config eksperimen (YAML)
│   └── transformer_base.yaml
├── scraper/
│   ├── idx_scraper.py     unduh ringkasan saham harian
│   └── build_panel.py     gabung raw → panel bersih (parquet)
├── data/
│   ├── raw/               hasil scrape (gitignored)
│   └── processed/         panel.parquet (gitignored)
├── src/
│   ├── preprocess.py      raw → fitur (log return, volume, foreign flow, ...)
│   ├── dataset.py         windowing sequence untuk Transformer
│   ├── train_test.py      training loop + walk-forward backtest
│   ├── run_train_test.py  CLI entrypoint (config → hasil)
│   └── utils.py           config/seed/metrics
├── models/
│   ├── transformer.py     TransformerPolicy
│   └── checkpoints/       bobot tersimpan (gitignored)
├── results/               metrik & plot (gitignored)
└── logs/                  (gitignored)
```

Beda dari DLSA: **tidak ada** `factor_models/` dan `residuals/`.

### Aliran data

```
data/raw/  ──build_panel──▶  data/processed/panel.parquet  (long: date × ticker × fitur)
   panel  ──preprocess──▶  fitur ternormalisasi (fit stats di TRAIN saja)
   fitur  ──dataset──▶  sampel (T_lookback × F) per (ticker, hari)
  sampel  ──TransformerPolicy──▶  posisi per aset  ──portfolio──▶  PnL / Sharpe
```

### Panel & fitur

- **Panel** (`build_panel.py`): long-format indeks `(date, ticker)`; kolom
  OHLC, prev_close, volume, value, frequency, foreign buy/sell, listed shares.
  Bersihkan tipe numerik, buang instrumen non-saham (rights/warrant) bila mau,
  buang hari volume nol (suspend).
- **Fitur** (`preprocess.py`), semua kausal (tanpa look-ahead):
  log return, log volume/value, high-low range, rasio net foreign flow,
  rolling z-score. Normalisasi **cross-sectional per hari** (gaya DLSA),
  stats di-fit **hanya di window train** untuk cegah kebocoran.
- **Windowing** (`dataset.py`): jendela lookback `T` (mis. 60 hari bursa) →
  target return periode berikutnya; masking untuk aset yang hilang/suspend agar
  batch cross-section tetap rapi. Split **walk-forward by date**, bukan acak.

### Arsitektur model (`models/transformer.py`)

`TransformerPolicy`: `Linear(F→d_model)` + positional encoding →
`N × TransformerEncoderLayer` (self-attention sepanjang lookback) →
pooling (last/mean) → head `Linear(d_model→1)` → `tanh` (posisi di [-1, 1]).
Konstruksi portofolio (dollar-neutral / batas leverage) dilakukan di
`train_test.py`, di luar modul model.

---

## Bagian 3 — Adaptasi DLSA: raw data → Transformer (tanpa factor model)

**DLSA asli:**
`raw → factor model (PCA/IPCA/Fama-French) → residual → CNN+Transformer policy → PnL`.

**Versi kita:**
`raw IDX → fitur ternormalisasi → Transformer policy → PnL`.

Yang **dibuang**: seluruh pipeline factor model & residual (`run_factor_model.py`,
`factor_models/`, `residuals/`).

Yang **dipertahankan** dari kerangka DLSA:
- Pemisahan `train_test.py` (logika) vs `run_train_test.py` (config/logging/save).
- Config YAML per eksperimen di `configs/`.
- **Objective berbasis trading**: loss = negatif Sharpe/mean return portofolio,
  dengan penalti turnover + asumsi biaya transaksi (bps), constraint
  dollar-neutral / leverage.
- Evaluasi **walk-forward** yang ketat waktu.

**Konsekuensi & catatan riset:**
- Tanpa factor model, Transformer menanggung beban belajar co-movement lintas
  aset sendiri → butuh normalisasi cross-sectional yang kuat + regularisasi.
- Awasi look-ahead pada normalisasi & split (paling rawan bug).
- Biaya transaksi IDX relatif tinggi → penalti turnover penting agar hasil
  tidak over-optimistis.

---

## Roadmap & urutan kerja

1. **Repo & scaffold** — selesai (file ini + skeleton + config).
2. **Scraper** — implement `idx_scraper.py`, uji ambil beberapa hari, lalu full 5 thn.
3. **build_panel** — validasi skema, hasilkan `panel.parquet`, cek kualitas data.
4. **preprocess + dataset** — fitur, normalisasi anti-leakage, windowing, split.
5. **Model + training** — `TransformerPolicy`, loop train, early-stop by val Sharpe.
6. **Backtest** — walk-forward, metrik (Sharpe, return, vol, max drawdown, turnover).
7. **Baseline pembanding** — mis. reversal sederhana / buy-and-hold IHSG.
8. **Verifikasi** — unit test anti look-ahead, smoke test end-to-end di data kecil.

## Pertanyaan terbuka

- Konfirmasi endpoint & nama field IDX terkini (bisa berubah).
- ~~Perlakuan aksi korporasi (stock split, dividen) → penyesuaian harga?~~
  **Selesai (split):** return harian pakai `close / Previous`; IDX sudah
  menyesuaikan `Previous` di ex-date, jadi split tidak jadi return palsu
  (lihat `preprocess.adjusted_log_return`). Dividen tunai TIDAK disesuaikan
  (data ringkasan tidak memuat dividen) → return = price return.
- ~~Threshold likuiditas~~ **Selesai:** blok config `universe.min_value_idr`
  (median nilai transaksi 20 hari ≥ ambang, kausal) di `src/market.py`.
- ~~Handling survivorship bias~~ **Selesai:** panel dari snapshot harian sudah
  memuat emiten yang belakangan delisting; backtest menulis turun posisi yang
  hilang dari panel (`portfolio.delist_after/delist_return`).

## Update 2026-07-04 — lapisan realisme long-only

Temuan review: backtest lama menggelembungkan hasil. Perbaikan yang sudah
diimplementasikan (lihat juga `tests/`):

1. **Aksi korporasi**: return via `Previous` (adjusted), path log-price
   ter-adjust untuk momentum & target. Sebelumnya split 1:10 terbaca -90%.
2. **Gap/suspensi**: target hanya terdefinisi bila t+1 adalah hari bursa
   PERSIS berikutnya dan sahamnya benar-benar ditransaksikan; sampel hari-t
   butuh `valid_day` (traded + lolos screen likuiditas).
3. **Backtest stateful** (`src/backtest.py`): tidak bisa beli saham yang
   mentok ARA (offer=0), tidak bisa jual yang mentok ARB (bid=0) — posisi
   "nyangkut" tetap dipegang dan menanggung return berikutnya; biaya beli/jual
   terpisah (komisi + pajak jual 0,1%); kas idle dapat bunga `rf_annual`;
   emiten yang hilang dari panel ditulis turun.
4. **Objective selaras dengan yang ditradingkan** (`train_dlsa`): blok
   hari BERURUTAN, turnover dikenai biaya di dalam loss (net Sharpe, excess
   rf); `softmax_temp` 0.1 supaya bobot terkonsentrasi seperti buku top-N;
   slot kas opsional (`allow_cash`); early stop pakai net Sharpe aturan top-N
   di validasi, bukan portofolio softmax.
5. **Walk-forward beneran**: `split.mode: walk_forward` — retrain per fold
   (default 6 bulan), skor test dirangkai lintas fold sebelum backtest.
6. **Metrik**: Sharpe dihitung EXCESS rf (~5,5%); plus alpha/beta/IR terhadap
   proxy IHSG (long-only didominasi beta).
7. **Anti look-ahead**: unit test di `tests/test_preprocess.py` (mutasi data
   masa depan tidak boleh mengubah fitur ≤ t).
