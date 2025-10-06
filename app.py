import io
import os
import base64
import time
import json
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Gig 2: Python Analytics Dashboard", layout="wide")

# ---- Sidebar: Branding / Mode ----
st.sidebar.title("Gig 2 • Python Dashboard")
mode = st.sidebar.radio("Mode", ["Portfolio Demo (sample data)", "My Data (upload)"])
st.sidebar.markdown("---")
st.sidebar.subheader("Branding")
brand_name = st.sidebar.text_input("Brand/Client Name", "Bait N' Shack Analytics")
accent = st.sidebar.color_picker("Accent color", "#2C7BE5")
st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit • Export insights as CSV & PNG")

st.caption("Note: If the bundled sample CSV is missing on the deploy platform, a synthetic sample dataset will be generated automatically.")

# ---- Helper: CSS for accent ----
st.markdown(f"""
    <style>
      :root {{
        --accent: {accent};
      }}
      .metric-card {{
        border: 1px solid #e6e6e6;
        border-left: .4rem solid var(--accent);
        padding: .8rem 1rem;
        border-radius: .5rem;
        background: #fff;
      }}
      .small-note {{
        font-size: 0.8rem;
        color: #666;
      }}
    </style>
""", unsafe_allow_html=True)

# ---- Data ingestion ----
@st.cache_data
def load_sample():
    sample_path = os.path.join("data", "sample_sales.csv")
    # If a bundled sample exists, use it; otherwise generate a synthetic one
    try:
        if os.path.exists(sample_path):
            return pd.read_csv(sample_path, parse_dates=["order_date"])  # type: ignore[arg-type]
    except Exception:
        pass

    # --- Fallback synthetic dataset (works on Streamlit Cloud too) ---
    rng = pd.date_range("2024-01-01", periods=180, freq="D")
    rng = rng.append(pd.date_range("2024-07-01", periods=90, freq="D"))
    n = len(rng)
    np.random.seed(42)
    cats = np.random.choice(["Online", "Retail", "Wholesale"], size=n, p=[0.5, 0.35, 0.15])
    regions = np.random.choice(["East", "South", "Midwest", "West"], size=n)
    units = np.random.poisson(lam=8, size=n).clip(min=0)
    price = np.random.lognormal(mean=3.0, sigma=0.3, size=n).round(2)
    sales = (units * price).round(2)
    df_gen = pd.DataFrame({
        "order_date": rng,
        "channel": cats,
        "region": regions,
        "units": units,
        "price": price,
        "sales": sales,
    })
    return df_gen

def load_user_file(uploaded):
    # Try UTF-8 first, then fallback to common encodings
    # Streamlit uploader provides a file-like object compatible with pandas
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            dfu = pd.read_csv(uploaded, encoding=enc)
            break
        except Exception:
            uploaded.seek(0)
            dfu = None
    if dfu is None:
        # Last resort: let pandas sniff; if still failing, raise
        uploaded.seek(0)
        dfu = pd.read_csv(uploaded, engine="python", error_bad_lines=False)  # type: ignore[call-arg]

    # Best-effort parse of columns containing 'date' or 'time'
    for col in dfu.columns:
        cl = str(col).lower()
        if "date" in cl or "time" in cl or "dt" in cl:
            try:
                dfu[col] = pd.to_datetime(dfu[col], errors="ignore")
            except Exception:
                pass
    return dfu

if mode == "Portfolio Demo (sample data)":
    df = load_sample()
    st.toast("Loaded sample dataset (bundled or generated).", icon="✅")
else:
    up = st.file_uploader("Upload a CSV (<= 50 MB)", type=["csv"])
    if up is not None:
        df = load_user_file(up)
        # Best-effort date parse
        for col in df.columns:
            if "date" in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    pass
        st.toast("Your dataset is loaded.", icon="✅")
    else:
        st.stop()

# ---- Basic data profile ----
st.title(f"{brand_name} • Analytics Overview")
st.write("Quick EDA, light cleaning helpers, and ready-to-export visuals for your report.")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Rows", f"{len(df):,}")
    st.markdown('</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Columns", f"{df.shape[1]:,}")
    st.markdown('</div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    missing = df.isna().sum().sum()
    pct_missing = (missing / (len(df)*df.shape[1]))*100 if len(df)>0 else 0
    st.metric("Missing Cells", f"{missing:,}", f"{pct_missing:.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Numeric Columns", f"{df.select_dtypes(include=[np.number]).shape[1]}")
    st.markdown('</div>', unsafe_allow_html=True)

with st.expander("Peek at the data"):
    st.dataframe(df.head(20), use_container_width=True)

with st.expander("Column Summary"):
    info = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(t) for t in df.dtypes],
        "n_unique": [df[c].nunique(dropna=False) for c in df.columns],
        "missing": [df[c].isna().sum() for c in df.columns]
    })
    st.dataframe(info, use_container_width=True)

# ---- Light cleaning helpers ----
st.subheader("🧹 Quick Cleaning")
cA, cB, cC = st.columns(3)

with cA:
    drop_dupes = st.checkbox("Drop duplicate rows")
with cB:
    fill_na_mode = st.checkbox("Fill missing categorical values with mode")
with cC:
    fill_na_zero = st.checkbox("Fill missing numeric values with 0")

df_clean = df.copy()
if drop_dupes:
    df_clean = df_clean.drop_duplicates()
if fill_na_mode:
    for col in df_clean.select_dtypes(include=["object","category"]).columns:
        if df_clean[col].isna().any():
            try:
                mode_val = df_clean[col].mode(dropna=True)[0]
                df_clean[col] = df_clean[col].fillna(mode_val)
            except Exception:
                pass
if fill_na_zero:
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        if df_clean[col].isna().any():
            df_clean[col] = df_clean[col].fillna(0)

if not df_clean.equals(df):
    st.success("Cleaning applied. Visuals below use the cleaned data. You can also download it.")
    st.download_button("Download cleaned CSV", data=df_clean.to_csv(index=False), file_name="cleaned_data.csv")

# ---- Visuals ----
st.subheader("📊 Visuals")

# Choose a date column (optional)
date_cols = [c for c in df_clean.columns if "date" in c.lower()] + list(df_clean.select_dtypes(include=["datetime64[ns]"]).columns)
date_col = st.selectbox("Optional: choose a date/time column for time-series", options=["(none)"] + date_cols)

# Choose metrics & dimensions
num_cols = list(df_clean.select_dtypes(include=[np.number]).columns)
cat_cols = list(df_clean.select_dtypes(exclude=[np.number, "datetime64[ns]"]).columns)

metric = st.selectbox("Metric (numeric)", options=num_cols if num_cols else ["<no numeric columns>"])
dim = st.selectbox("Dimension (categorical)", options=cat_cols if cat_cols else ["<no categorical columns>"])

# Aggregation
agg_fn = st.selectbox("Aggregation", options=["sum","mean","median","count","min","max"])

def agg_data(dfX, by, y, fn):
    if by == "<no categorical columns>" or y == "<no numeric columns>":
        return None
    if fn == "sum":
        return dfX.groupby(by, dropna=False)[y].sum().reset_index()
    if fn == "mean":
        return dfX.groupby(by, dropna=False)[y].mean().reset_index()
    if fn == "median":
        return dfX.groupby(by, dropna=False)[y].median().reset_index()
    if fn == "count":
        return dfX.groupby(by, dropna=False)[y].count().reset_index()
    if fn == "min":
        return dfX.groupby(by, dropna=False)[y].min().reset_index()
    if fn == "max":
        return dfX.groupby(by, dropna=False)[y].max().reset_index()

# Bar chart
aggdf = agg_data(df_clean, dim, metric, agg_fn)
if aggdf is not None and len(aggdf) > 0:
    st.markdown("**Bar Chart**")
    fig1, ax1 = plt.subplots()
    ax1.bar(aggdf[dim].astype(str), aggdf[metric])
    ax1.set_xlabel(dim)
    ax1.set_ylabel(f"{agg_fn}({metric})")
    ax1.set_title(f"{agg_fn.upper()} of {metric} by {dim}")
    plt.xticks(rotation=20, ha="right")
    st.pyplot(fig1)
    buf = io.BytesIO()
    fig1.savefig(buf, format="png", bbox_inches="tight")
    st.download_button("Download bar chart PNG", data=buf.getvalue(), file_name="bar_chart.png")

# Time series
if date_col != "(none)" and metric in df_clean.columns:
    if not np.issubdtype(df_clean[date_col].dtype, np.datetime64):
        try:
            ts = pd.to_datetime(df_clean[date_col])
        except Exception:
            ts = None
    else:
        ts = df_clean[date_col]

    if ts is not None:
        st.markdown("**Time Series**")
        tsdf = df_clean[[date_col, metric]].copy()
        tsdf[date_col] = pd.to_datetime(tsdf[date_col], errors="coerce")
        tsdf = tsdf.dropna(subset=[date_col])
        tsdf = tsdf.sort_values(by=date_col)
        # Resample to weekly if high frequency
        try:
            tsdf = tsdf.set_index(date_col).resample("W")[metric].sum().reset_index()
        except Exception:
            pass
        fig2, ax2 = plt.subplots()
        ax2.plot(tsdf[date_col], tsdf[metric])
        ax2.set_xlabel(date_col)
        ax2.set_ylabel(metric)
        ax2.set_title(f"Weekly {metric} Trend")
        st.pyplot(fig2)
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format="png", bbox_inches="tight")
        st.download_button("Download time-series PNG", data=buf2.getvalue(), file_name="timeseries.png")

# Pivot-style table
st.subheader("📈 Pivot Table (Wide)")
if date_col != "(none)" and dim in df_clean.columns and metric in df_clean.columns:
    pivot_period = st.selectbox("Pivot period for date", ["M", "W", "Q", "Y"], index=0)
    tmp = df_clean[[date_col, dim, metric]].copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col])
    tmp["period"] = tmp[date_col].dt.to_period(pivot_period).dt.to_timestamp()
    piv = tmp.pivot_table(index=dim, columns="period", values=metric, aggfunc="sum").fillna(0)
    st.dataframe(piv, use_container_width=True)
    csv = piv.reset_index().to_csv(index=False)
    st.download_button("Download pivot CSV", data=csv, file_name="pivot_table.csv")
else:
    st.info("Select a date column, a categorical dimension, and a numeric metric to view the pivot table.")

# ---- Notes / Export ----
st.subheader("📝 Notes")
notes = st.text_area("Executive notes / observations", placeholder="Add bullets or commentary here...")
if notes:
    st.download_button("Download notes (.txt)", data=notes, file_name="notes.txt")

st.caption("© 2025 Luke Medlin • Demo dashboard for Fiverr Gig 2 (Python-based analytics dashboard).")
