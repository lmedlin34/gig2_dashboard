# Gig 2 — Python Analytics Dashboard (Streamlit)

A ready-to-run Streamlit app you can showcase on your Fiverr profile.

## What it does
- Upload CSV **or** use included sample data (Portfolio Demo mode)
- KPI cards: rows, columns, missing cells, numeric columns
- Light cleaning toggles: drop duplicates, fill categorical NA with mode, fill numeric NA with 0
- Charts: **Bar** (metric by dimension) and **Weekly time series** (if a date column exists)
- Pivot-style wide table with monthly/weekly/quarterly/yearly columns
- Download cleaned data, charts (PNG), pivot (CSV), and notes (TXT)

## Quickstart
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open the local URL Streamlit provides.

## Files
- `app.py` — main app
- `data/sample_sales.csv` — demo dataset for instant preview
- `.streamlit/config.toml` — theme tweaks
- `requirements.txt` — minimal dependencies

## Customize
- Add your logo in the sidebar (code section could be extended)
- Adjust color defaults in `app.py` (accent color picker is already included)
- Swap in your own sample dataset(s) in `/data`

---
(c) 2025 Luke Medlin