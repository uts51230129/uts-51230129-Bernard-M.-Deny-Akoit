from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import secrets
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, send_file, session, url_for
from werkzeug.utils import secure_filename

import charts as chartkit

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
ALLOWED_EXTENSIONS = {".xlsx", ".xls", ".csv"}

# Per-process in-memory workspace. Intended for local single-user academic use.
WORKSPACES: dict[str, dict] = {}

NAV = [
    ("overview", "Overview", "layout-dashboard", "Executive overview and process status"),
    ("dataset", "Dataset Management", "database", "Upload, inspect, and validate source data"),
    ("eda", "Data Profiling & EDA", "chart-no-axes-combined", "Profiling, distributions, and data quality"),
    ("preprocessing", "Data Preprocessing", "sliders-horizontal", "Cleaning rules and transformation log"),
    ("validation", "Data Validation & Audit", "shield-check", "Business-rule checks and audit trail"),
    ("apriori", "Market Basket Analysis", "network", "Frequent itemsets and association rules"),
    ("segmentation", "Customer Segmentation", "chart-scatter", "RFM features and K-Means clusters"),
    ("bi", "Business Intelligence", "chart-no-axes-combined", "Interactive retail performance dashboard"),
    ("recommendations", "Business Recommendations", "lightbulb", "Evidence-based recommendations"),
    ("reports", "Reports & Export", "file-chart-column", "Download Excel and PDF reports"),
    ("system", "System Information", "settings", "Methodology, limitations, and system status"),
    ("pengembang", "Pengembang", "user-round", "Profil pengembang aplikasi"),
]

def json_safe(value):
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)): return value.isoformat()
    if pd.isna(value): return None
    return value

def clean_json_records(df, limit=200):
    if df is None or df.empty: return []
    out = df.head(limit).copy()
    out = out.replace({np.nan: None})
    def _safe(v):
        if isinstance(v, (frozenset, set)):
            return " + ".join(sorted(map(str, v)))
        if isinstance(v, (list, tuple)):
            return [_safe(x) for x in v]
        return json_safe(v)
    return [{str(k): _safe(v) for k, v in row.items()} for row in out.to_dict("records")]

def workspace():
    sid = session.get("sid")
    if not sid:
        sid = secrets.token_urlsafe(16)
        session["sid"] = sid
    return WORKSPACES.setdefault(sid, {"source_name": None, "raw": None, "sheets": [],
        "active_sheet": None, "clean": None, "profiling": None, "quality": None,
        "log": [], "basket": None, "itemsets": None, "rules": None, "rfm": None,
        "cluster_profile": None, "clustered": None, "kmeans_meta": None,
        "results": None, "status": {}, "recommendations": [], "created_at": None,
        "apriori_ready": None, "apriori_meta": None, "validation_audit": None,
        "charts": {}})

def allowed(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def read_uploaded(file_storage):
    filename = secure_filename(file_storage.filename or "dataset.xlsx")
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(file_storage)
        return df, [], None
    xl = pd.ExcelFile(file_storage)
    sheets = xl.sheet_names
    preferred = "Data_Raw" if "Data_Raw" in sheets else sheets[0]
    df = pd.read_excel(xl, sheet_name=preferred)
    return df, sheets, preferred

def find_col(df, candidates):
    norm = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in df.columns}
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]", "", candidate.lower())
        if key in norm: return norm[key]
    return None

def resolve_columns(df):
    return {
        "transaction": find_col(df, ["Transaction_ID", "InvoiceNo", "TransactionID", "Order_ID"]),
        "date": find_col(df, ["Transaction_Date", "InvoiceDate", "Order_Date", "Date"]),
        "customer": find_col(df, ["Customer_ID", "CustomerID", "Customer_Id"]),
        "product_id": find_col(df, ["Product_ID", "StockCode", "ProductID", "SKU"]),
        "product": find_col(df, ["Product_Name", "Description", "Product", "Item_Name"]),
        "category": find_col(df, ["Category", "Product_Category"]),
        "quantity": find_col(df, ["Quantity", "Qty"]),
        "price": find_col(df, ["Unit_Price", "UnitPrice", "Price"]),
        "discount": find_col(df, ["Discount_Rate", "Discount", "DiscountRate"]),
        "total": find_col(df, ["Line_Total", "Total", "Sales", "LineTotal", "Revenue"]),
        "region": find_col(df, ["Region", "Country", "City"]),
        "payment": find_col(df, ["Payment_Method", "PaymentMethod", "Payment"]),
    }

def money(v):
    try: return "Rp " + f"{float(v):,.0f}".replace(",", ".")
    except Exception: return "—"

def thousands(v, suffix=""):
    """Format big numbers with Indonesian thousand separator (titik), e.g. 18.513."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—" if v is None else f"{v}{suffix}"
    if f is None or (isinstance(f, float) and not math.isfinite(f)):
        return "—"
    if f == int(f):
        return f"{int(f):,}".replace(",", ".") + suffix
    return f"{f:,.2f}".replace(",", ".") + suffix

@app.template_filter("thousand")
def thousand_filter(v):
    return thousands(v)

def profile_data(df):
    cols = []
    for c in df.columns:
        s = df[c]
        cols.append({"column": str(c), "dtype": str(s.dtype), "non_null": int(s.notna().sum()),
                     "missing": int(s.isna().sum()), "missing_pct": round(float(s.isna().mean()*100), 2),
                     "unique": int(s.nunique(dropna=True))})
    num = df.select_dtypes(include=np.number)
    stats = []
    if not num.empty:
        desc = num.describe().T.reset_index(names="column")
        stats = clean_json_records(desc, 100)
    return {"rows": int(len(df)), "columns": int(len(df.columns)), "column_profile": cols,
            "numeric_stats": stats, "duplicate_exact": int(df.duplicated().sum())}

def preprocess(df):
    """Conservative, auditable cleaning. Source frame remains unchanged."""
    raw = df.copy(deep=True)
    out = df.copy(deep=True)
    cols = resolve_columns(out)
    log = []
    log.append({"step":"Salinan kerja","before":len(out),"after":len(out),
                "action":"Data_Raw tidak dimodifikasi; pembersihan dilakukan pada salinan."})

    # Normalize whitespace in string columns; preserve missing values.
    before = len(out)
    text_cols = out.select_dtypes(include=["object", "string"]).columns
    changed = 0
    for c in text_cols:
        old = out[c].copy()
        out[c] = out[c].map(lambda x: re.sub(r"\s+", " ", x.strip()) if isinstance(x, str) else x)
        changed += int((old.fillna("__NA__") != out[c].fillna("__NA__")).sum())
    log.append({"step":"Standardisasi spasi teks","before":before,"after":len(out),
                "action":f"Trim dan rapikan spasi pada kolom teks; {changed} nilai berubah."})

    # Case-normalize known categorical columns while retaining readable title case.
    for key in ["category", "region", "payment"]:
        c = cols.get(key)
        if c and c in out.columns:
            out[c] = out[c].map(lambda x: x.title() if isinstance(x, str) else x)
    log.append({"step":"Standardisasi kategori","before":len(out),"after":len(out),
                "action":"Normalisasi kapitalisasi pada kategori, wilayah, dan metode pembayaran yang terdeteksi."})

    # Parse date
    if cols["date"]:
        c = cols["date"]
        parsed = pd.to_datetime(out[c], errors="coerce", dayfirst=False)
        invalid = int((out[c].notna() & parsed.isna()).sum())
        out[c] = parsed
        log.append({"step":"Validasi tanggal","before":len(out),"after":len(out),
                    "action":f"Konversi tanggal; {invalid} nilai tanggal non-kosong gagal diparse."})

    # Numeric coercion
    for key in ["quantity", "price", "discount", "total"]:
        c = cols.get(key)
        if c:
            before_invalid = int((out[c].notna() & pd.to_numeric(out[c], errors="coerce").isna()).sum())
            out[c] = pd.to_numeric(out[c], errors="coerce")
            log.append({"step":f"Konversi numerik: {c}","before":len(out),"after":len(out),
                        "action":f"Konversi ke numerik; {before_invalid} nilai tidak dapat dikonversi menjadi angka."})

    # Drop only exact duplicate rows, not transaction IDs.
    duplicate_mask = out.duplicated(keep="first")
    duplicate_count = int(duplicate_mask.sum())
    out = out.loc[~duplicate_mask].copy()
    log.append({"step":"Hapus duplikasi identik","before":len(raw),"after":len(out),
                "action":f"Menghapus {duplicate_count} baris duplikat yang identik; tidak menghapus berdasarkan Transaction_ID saja."})

    # Remove rows only where core transaction/product identifiers are missing.
    required = [cols[k] for k in ["transaction", "product"] if cols.get(k)]
    if not required and cols.get("product_id"): required = [cols["product_id"]]
    if required:
        mask = out[required].notna().all(axis=1)
        removed = int((~mask).sum())
        out = out.loc[mask].copy()
        log.append({"step":"Validasi identitas transaksi/produk","before":len(raw)-duplicate_count,"after":len(out),
                    "action":f"Mengeluarkan {removed} baris tanpa identitas inti yang terdeteksi: {', '.join(map(str, required))}."})

    # Mark invalid numeric rows rather than silently deleting them.
    invalid_count = 0
    for key in ["quantity", "price"]:
        c = cols.get(key)
        if c:
            invalid_count += int((out[c].notna() & (out[c] <= 0)).sum())
    log.append({"step":"Audit nilai numerik","before":len(out),"after":len(out),
                "action":f"{invalid_count} kejadian Quantity/Unit Price nol atau negatif ditandai untuk audit; tidak otomatis dihapus."})

    # Validate/recalculate line total only where all formula inputs are valid.
    q, p, d, t = cols["quantity"], cols["price"], cols["discount"], cols["total"]
    if q and p and t:
        valid = out[q].notna() & out[p].notna() & (out[q] > 0) & (out[p] > 0)
        if d:
            disc = out[d].copy()
            # Treat values >1 and <=100 as percent, but document it; preserve original values in raw.
            disc_fraction = disc.where(disc <= 1, disc / 100.0)
            valid &= disc_fraction.notna() & disc_fraction.between(0, 1)
            expected = out[q] * out[p] * (1 - disc_fraction)
        else:
            expected = out[q] * out[p]
        mismatch = valid & out[t].notna() & ~np.isclose(out[t].astype(float), expected.astype(float), rtol=0.015, atol=1.0, equal_nan=False)
        log.append({"step":"Validasi rumus Line Total","before":len(out),"after":len(out),
                    "action":f"{int(mismatch.sum())} nilai berbeda dari rumus estimasi pada baris dengan komponen valid. Nilai tidak ditimpa otomatis."})
    else:
        log.append({"step":"Validasi rumus Line Total","before":len(out),"after":len(out),
                    "action":"Tidak dilakukan: kolom Quantity, Unit Price, atau Line Total tidak terdeteksi."})

    return out.reset_index(drop=True), log, cols

def build_validation_audit(raw, clean, cols):
    """Rule-based audit trail. Findings are recorded, never silently deleted."""
    rows = []
    def add(name, severity, before, after, note):
        rows.append({"check": name, "severity": severity, "count": int(before or 0),
                     "count_before": int(before or 0), "count_after": int(after or 0), "note": note})

    dup_before = int(raw.duplicated().sum())
    dup_after = int(clean.duplicated().sum())
    add("Baris duplikat identik", "Tinggi" if dup_before else "Info", dup_before, dup_after,
        "Duplikat identik dihapus pada data bersih; ID transaksi saja tidak dihapus.")

    for key in ["transaction", "customer", "product", "product_id"]:
        c = cols.get(key)
        if not c: continue
        b = int(raw[c].isna().sum()); a = int(clean[c].isna().sum())
        sev = "Tinggi" if (key in ("transaction", "product") and b) else ("Sedang" if b else "Info")
        add(f"Missing · {c}", sev, b, a,
            "Kolom inti transaksi/produk wajib terisi." if key in ("transaction", "product")
            else "Missing value dicatat, tidak diisi otomatis.")

    if cols.get("date"):
        c = cols["date"]
        parsed = pd.to_datetime(raw[c], errors="coerce")
        invalid = int((raw[c].notna() & parsed.isna()).sum())
        future = int((parsed > pd.Timestamp.now()).sum())
        clean_parsed = pd.to_datetime(clean[c], errors="coerce") if c in clean.columns else None
        invalid_after = int((clean[c].notna() & clean_parsed.isna()).sum()) if clean_parsed is not None else invalid
        add("Tanggal tidak valid", "Tinggi" if invalid else "Info", invalid, invalid_after,
            "Nilai tanggal yang gagal diparse ditandai untuk audit.")
        add("Tanggal di masa depan", "Sedang" if future else "Info", future, future,
            "Tanggal setelah waktu sistem saat ini perlu ditinjau.")

    for key, severity_bad, label in [("quantity", "Tinggi", "Quantity nol/negatif"),
                                     ("price", "Tinggi", "Unit price nol/negatif"),
                                     ("discount", "Sedang", "Diskon di luar rentang [0,1]")]:
        c = cols.get(key)
        if not c: continue
        s = pd.to_numeric(raw[c], errors="coerce")
        if key == "discount":
            frac = s.where(s <= 1, s / 100.0)
            bad = int((s.notna() & ~frac.between(0, 1)).sum())
        else:
            bad = int((s.notna() & (s <= 0)).sum())
        cs = pd.to_numeric(clean[c], errors="coerce")
        if key == "discount":
            cfrac = cs.where(cs <= 1, cs / 100.0)
            bad_after = int((cs.notna() & ~cfrac.between(0, 1)).sum())
        else:
            bad_after = int((cs.notna() & (cs <= 0)).sum())
        add(label, severity_bad if bad else "Info", bad, bad_after,
            "Nilai ditandai untuk audit; tidak otomatis dihapus.")

    q, p, d, t = cols.get("quantity"), cols.get("price"), cols.get("discount"), cols.get("total")
    if q and p and t:
        s = raw
        valid = pd.to_numeric(s[q], errors="coerce").notna() & pd.to_numeric(s[p], errors="coerce").notna()
        expected = pd.to_numeric(s[q], errors="coerce") * pd.to_numeric(s[p], errors="coerce")
        if d:
            disc = pd.to_numeric(s[d], errors="coerce")
            frac = disc.where(disc <= 1, disc / 100.0)
            expected = expected * (1 - frac)
        mismatch = int((valid & pd.to_numeric(s[t], errors="coerce").notna()
                        & ~np.isclose(pd.to_numeric(s[t], errors="coerce").astype(float), expected.astype(float),
                                      rtol=0.015, atol=1.0, equal_nan=False)).sum())
        cvalid = pd.to_numeric(clean[q], errors="coerce").notna() & pd.to_numeric(clean[p], errors="coerce").notna()
        cexpected = pd.to_numeric(clean[q], errors="coerce") * pd.to_numeric(clean[p], errors="coerce")
        if d:
            cdisc = pd.to_numeric(clean[d], errors="coerce")
            cfrac = cdisc.where(cdisc <= 1, cdisc / 100.0)
            cexpected = cexpected * (1 - cfrac)
        cmismatch = int((cvalid & pd.to_numeric(clean[t], errors="coerce").notna()
                         & ~np.isclose(pd.to_numeric(clean[t], errors="coerce").astype(float), cexpected.astype(float),
                                       rtol=0.015, atol=1.0, equal_nan=False)).sum())
        add("Line Total ≠ Quantity × Harga × (1−Diskon)", "Tinggi" if mismatch else "Info", mismatch, cmismatch,
            "Ketidakcocokan rumus ditandai; nilai asli dipertahankan untuk ditinjau.")
    return rows

def build_apriori_readiness(clean, cols):
    """Compute the exact numbers a user needs before running Apriori."""
    tx, prod = cols.get("transaction"), cols.get("product")
    if clean is None or not tx or not prod:
        return None
    tmp = clean[[tx, prod]].dropna().drop_duplicates()
    baskets = int(tmp[tx].nunique())
    products = int(tmp[prod].nunique())
    items_per = tmp.groupby(tx)[prod].nunique()
    avg_items = float(items_per.mean()) if baskets else 0.0
    max_items = int(items_per.max()) if baskets else 0
    rows = int(len(tmp))
    suggested_support = round(min(0.2, max(0.005, 3.0 / baskets)), 4) if baskets else 0.02
    return {"transactions": baskets, "products": products, "rows": rows,
            "avg_items": round(avg_items, 2), "max_items": max_items,
            "suggested_support": suggested_support, "suggested_confidence": 0.3,
            "support_note": f"≈ {thousands(round(suggested_support * baskets))} transaksi per itemset",
            "ready": baskets >= 2 and products >= 2}

def calculate_analysis(w):
    raw = w["raw"]
    clean, log, cols = preprocess(raw)
    w["clean"], w["log"], w["columns"] = clean, log, cols
    w["profiling"] = profile_data(raw)
    w["quality"] = profile_data(clean)
    w["validation_audit"] = build_validation_audit(raw, clean, cols)
    w["apriori_ready"] = build_apriori_readiness(clean, cols)
    status = {"eda":"success", "preprocessing":"success", "validation":"success",
              "apriori":"not_run", "kmeans":"not_run", "bi":"success", "recommendations":"pending"}
    # Net sales: use line total when available; only valid positive transaction lines.
    total_col, qty_col, price_col = cols["total"], cols["quantity"], cols["price"]
    if total_col:
        sales_series = pd.to_numeric(clean[total_col], errors="coerce")
        if qty_col: sales_series = sales_series.where(pd.to_numeric(clean[qty_col], errors="coerce") > 0)
        if price_col: sales_series = sales_series.where(pd.to_numeric(clean[price_col], errors="coerce") > 0)
        net_sales = float(sales_series.sum(min_count=1) or 0)
    elif qty_col and price_col:
        net_sales = float((clean[qty_col] * clean[price_col]).sum())
    else:
        net_sales = None
    tx_col, cust_col, prod_col = cols["transaction"], cols["customer"], cols["product"]
    transactions = int(clean[tx_col].nunique()) if tx_col else None
    customers = int(clean[cust_col].nunique()) if cust_col else None
    if cust_col: customers = int(clean[cust_col].nunique(dropna=True))
    basket_size = None
    if tx_col and prod_col:
        basket_size = float(clean.groupby(tx_col)[prod_col].nunique().mean()) if len(clean) else 0.0
    qty_sold = float(clean[qty_col].sum()) if qty_col else None

    monthly = []
    if cols["date"] and total_col:
        temp = clean.dropna(subset=[cols["date"]]).copy()
        temp["_month"] = temp[cols["date"]].dt.to_period("M").astype(str)
        temp["_sales"] = pd.to_numeric(temp[total_col], errors="coerce")
        monthly = clean_json_records(temp.groupby("_month", as_index=False)["_sales"].sum().rename(columns={"_month":"month","_sales":"sales"}), 100)
    top_products = []
    if prod_col and total_col:
        top_products = clean_json_records(clean.groupby(prod_col, dropna=True)[total_col].sum().sort_values(ascending=False).head(10).rename("sales").reset_index().rename(columns={prod_col:"product"}), 10)
    categories = []
    if cols["category"] and total_col:
        categories = clean_json_records(clean.groupby(cols["category"], dropna=True)[total_col].sum().sort_values(ascending=False).head(12).rename("sales").reset_index().rename(columns={cols["category"]:"category"}), 20)

    w["results"] = {"source_name":w["source_name"], "raw_rows":len(raw), "clean_rows":len(clean),
        "raw_columns":len(raw.columns), "clean_columns":len(clean.columns), "net_sales":net_sales,
        "transactions":transactions, "customers":customers, "basket_size":basket_size,
        "quantity_sold":qty_sold, "monthly_sales":monthly, "top_products":top_products,
        "category_sales":categories, "columns":cols, "profile":w["profiling"], "quality":w["quality"],
        "log":log, "generated_at":datetime.now().isoformat(timespec="seconds"),
        "apriori_ready":w["apriori_ready"], "validation_audit":w["validation_audit"]}
    # Server-side charts for preprocessing, validation, and profiling pages.
    numeric_summary = []
    num_df = clean.select_dtypes(include=np.number)
    for c in num_df.columns:
        s = num_df[c].dropna()
        if s.empty: continue
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr = q3 - q1
        outliers = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        numeric_summary.append({"column": str(c), "count": int(s.size), "mean": float(s.mean()),
            "median": float(s.median()), "std": float(s.std()), "min": float(s.min()), "q1": q1,
            "q3": q3, "max": float(s.max()), "iqr": iqr, "outliers": outliers,
            "skewness": float(s.skew())})
    w["numeric_summary"] = numeric_summary
    charts = {}
    charts["rows_flow"] = chartkit.rows_flow(log)
    charts["missing_compare"] = chartkit.missing_before_after(w["profiling"], w["quality"])
    charts["numeric_boxplot"] = chartkit.numeric_boxplot(clean, log_scale=True)
    charts["numeric_boxplot_raw"] = chartkit.numeric_boxplot(raw, log_scale=True)
    charts["audit_severity"] = chartkit.audit_severity_chart(w["validation_audit"])
    charts["quantity_hist"] = chartkit.distribution_chart(
        clean[cols["quantity"]] if cols.get("quantity") else None,
        "Distribusi Quantity (data bersih)")
    charts["price_hist"] = chartkit.distribution_chart(
        clean[cols["price"]] if cols.get("price") else None,
        "Distribusi Unit Price (data bersih)", color="#14b8a6")
    w["charts"] = {k: v for k, v in charts.items() if v}
    w["status"] = status
    w["recommendations"] = build_recommendations(w)
    return w["results"]

def build_recommendations(w):
    r = w.get("results") or {}
    clean = w.get("clean")
    cols = w.get("columns", {})
    recs = []
    if r.get("top_products"):
        p = r["top_products"][0]
        share = ""
        if r.get("net_sales") and p.get("sales"):
            share = f" Kontribusi sekitar {p['sales']/r['net_sales']*100:.1f}% dari total net sales."
        recs.append({"priority":"Tinjau","title":"Evaluasi produk dengan penjualan tertinggi",
          "evidence":f"Produk teratas berdasarkan nilai penjualan tercatat: {p.get('product')} ({money(p.get('sales'))}).{share}",
          "action":"Periksa ketersediaan stok dan margin sebelum menentukan prioritas promosi.",
          "kpi":"Stock-out rate, net sales, gross margin","basis":"Agregasi penjualan produk dari data bersih"})
    if r.get("category_sales") and len(r["category_sales"]) >= 2:
        top_cat, second_cat = r["category_sales"][0], r["category_sales"][1]
        recs.append({"priority":"Tinjau","title":"Fokuskan assortmen pada kategori unggulan",
          "evidence":f"Kategori {top_cat.get('category')} memimpin ({money(top_cat.get('sales'))}), disusul {second_cat.get('category')} ({money(second_cat.get('sales'))}).",
          "action":"Alokasikan ruang rak dan anggaran pembelian ke kategori dengan kontribusi terbesar; tinjau kategori berkontribusi rendah.",
          "kpi":"Sales per kategori, inventory turnover","basis":"Agregasi kategori dari data bersih"})
    if w.get("rules") is not None and not w["rules"].empty:
        top = w["rules"].iloc[0]
        recs.append({"priority":"Peluang","title":"Uji cross-selling berdasarkan association rules",
          "evidence":f"Terdapat {len(w['rules'])} association rules. Rule terkuat: {top.get('antecedents_text')} → {top.get('consequents_text')} (lift {top.get('lift'):.2f}, confidence {top.get('confidence'):.0%}).",
          "action":"Uji bundling terbatas dan bandingkan hasil dengan kelompok kontrol.",
          "kpi":"Attach rate, basket value, conversion rate","basis":"Hasil Apriori aktual"})
    if w.get("cluster_profile") is not None and not w["cluster_profile"].empty:
        prof = w["cluster_profile"]
        best = prof.loc[prof["Monetary_Mean"].idxmax()] if "Monetary_Mean" in prof else None
        detail = f" Cluster dengan nilai rata-rata tertinggi: C{int(best['Cluster'])} ({money(best['Monetary_Mean'])}, {int(best['Customer_Count'])} pelanggan)." if best is not None else ""
        recs.append({"priority":"Peluang","title":"Rancang komunikasi per segmen pelanggan",
          "evidence":f"Segmentasi menghasilkan {len(prof)} cluster.{detail}",
          "action":"Sesuaikan kampanye dengan profil RFM tiap cluster dan evaluasi retensi.",
          "kpi":"Repeat purchase rate, retention rate","basis":"Profil RFM dan hasil K-Means aktual"})
    if clean is not None and cols.get("payment"):
        c = cols["payment"]
        top_pay = clean[c].dropna().value_counts()
        if len(top_pay):
            pct = top_pay.iloc[0] / max(top_pay.sum(), 1) * 100
            recs.append({"priority":"Tinjau","title":"Optimalkan kanal pembayaran populer",
              "evidence":f"Metode pembayaran {top_pay.index[0]} dipakai pada {pct:.1f}% baris transaksi bersih ({thousands(top_pay.iloc[0])} baris).",
              "action":"Negosiasikan biaya transaksi dan pastikan ketersediaan kanal tersebut di seluruh kasir.",
              "kpi":"Payment success rate, MDR cost","basis":"Distribusi metode pembayaran dari data bersih"})
    if w.get("validation_audit"):
        high = [a for a in w["validation_audit"] if a.get("severity") == "Tinggi" and a.get("count", 0) > 0]
        if high:
            recs.append({"priority":"Perbaikan","title":"Perbaiki kualitas data sebelum pelaporan final",
              "evidence":f"{len(high)} pemeriksaan audit tingkat tinggi masih menemukan masalah, contoh: {high[0].get('check')} ({thousands(high[0].get('count'))} temuan).",
              "action":"Perbaiki proses pencatatan di sumber sistem dan jalankan ulang preprocessing setelah perbaikan.",
              "kpi":"Missing rate, invalid value rate","basis":"Hasil validasi dan audit data"})
    if not recs:
        recs.append({"priority":"Menunggu hasil","title":"Lengkapi analisis sebelum menyimpulkan strategi",
          "evidence":"Belum ada hasil algoritma yang cukup untuk rekomendasi lintas-proses.",
          "action":"Jalankan Apriori dan K-Means setelah preprocessing dan validasi selesai.",
          "kpi":"Status proses dan kelengkapan analisis","basis":"Status sistem"})
    return recs

def _apriori_fallback(matrix, min_support, max_len=4):
    """Pure pandas Apriori used when mlxtend is unavailable."""
    from itertools import combinations
    n = len(matrix)
    cols = list(matrix.columns)
    freq = []
    # Level 1
    single = matrix.mean()
    current = {frozenset([c]): float(v) for c, v in single.items() if v >= min_support}
    freq.extend((k, v) for k, v in current.items())
    prev = list(current.keys())
    size = 2
    while prev and size <= max_len:
        candidates = {}
        prev_set = set(prev)
        for a, b in combinations(sorted(prev, key=lambda x: sorted(x)), 2):
            union = a | b
            if len(union) != size or union in candidates: continue
            if all(frozenset(sub) in prev_set for sub in combinations(union, size - 1)):
                candidates[union] = None
        nxt = []
        for union in candidates:
            mask = matrix[list(union)].all(axis=1)
            support = float(mask.mean())
            if support >= min_support:
                nxt.append(union)
                freq.append((union, support))
        prev = nxt
        size += 1
    if not freq:
        return pd.DataFrame(columns=["itemsets", "support"])
    return pd.DataFrame([{"itemsets": k, "support": v} for k, v in freq])

def _rules_fallback(itemsets, min_confidence):
    """Association rules from frequent itemsets without mlxtend."""
    from itertools import combinations
    sup = {frozenset(r["itemsets"]): float(r["support"]) for _, r in itemsets.iterrows()}
    rows = []
    for itemset, support in sup.items():
        if len(itemset) < 2: continue
        items = sorted(itemset)
        for r in range(1, len(items)):
            for ante in combinations(items, r):
                a = frozenset(ante); c = itemset - a
                if not c: continue
                sa, sc = sup.get(a), sup.get(c)
                if not sa or not sc: continue
                confidence = support / sa
                if confidence >= min_confidence:
                    rows.append({"antecedents": a, "consequents": c, "support": support,
                                 "confidence": confidence, "lift": confidence / sc})
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["antecedents", "consequents", "support", "confidence", "lift"])

def run_apriori(w, min_support=None, min_confidence=None):
    """Run Apriori with mlxtend when available, fallback engine otherwise.

    Empty results trigger an automatic threshold relaxation (documented) so the
    process always produces actionable output instead of a silent dead end.
    """
    engine = "mlxtend"
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
    except Exception:
        apriori, association_rules = None, None
        engine = "fallback-pandas"
    df = w.get("clean")
    cols = w.get("columns", {})
    tx, prod = cols.get("transaction"), cols.get("product")
    if df is None or not tx or not prod:
        raise ValueError("Apriori memerlukan kolom transaksi dan nama produk yang dapat dipetakan.")
    tmp = df[[tx, prod]].dropna().drop_duplicates()
    if tmp[tx].nunique() < 2: raise ValueError("Jumlah transaksi valid terlalu sedikit untuk Apriori.")
    matrix = pd.crosstab(tmp[tx], tmp[prod]).gt(0)
    if matrix.shape[1] < 2: raise ValueError("Diperlukan sedikitnya dua produk unik untuk Apriori.")
    if min_support is None:
        min_support = float(request.form.get("min_support", 0.02))
    if min_confidence is None:
        min_confidence = float(request.form.get("min_confidence", 0.3))
    if not 0 < min_support <= 1 or not 0 < min_confidence <= 1:
        raise ValueError("Support dan confidence harus berada di rentang (0, 1].")
    notes = []
    used_support, used_confidence = min_support, min_confidence
    if engine == "mlxtend":
        itemsets = apriori(matrix, min_support=used_support, use_colnames=True, max_len=4)
    else:
        itemsets = _apriori_fallback(matrix, used_support)
    # Relax support automatically if nothing is found.
    relaxed_to_floor = False
    for attempt in range(6):
        if not itemsets.empty: break
        if used_support <= 0.001:
            relaxed_to_floor = True
            break
        used_support = max(0.001, round(used_support / 2, 5))
        notes.append(f"Support diturunkan otomatis ke {used_support} karena tidak ada itemset pada ambang sebelumnya.")
        itemsets = apriori(matrix, min_support=used_support, use_colnames=True, max_len=4) if engine == "mlxtend" else _apriori_fallback(matrix, used_support)
    if not itemsets.empty and used_support != min_support:
        notes.append(f"Ambang support terpakai akhir: {used_support} (setara {thousands(round(used_support * matrix.shape[0]))} transaksi). Nilai awal yang disarankan sistem: {w.get('apriori_ready', {}).get('suggested_support', '—')}.")
    if relaxed_to_floor or itemsets.empty:
        notes.append(f"Tidak ada itemset frequent bahkan pada support 0.001. Periksa kolom produk/transaksi; nilai yang disarankan sistem untuk dataset ini: {w.get('apriori_ready', {}).get('suggested_support', '0.01')}.")
    if itemsets.empty:
        w["itemsets"] = pd.DataFrame(columns=["support", "itemsets", "items"])
        w["rules"] = pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift",
                                           "antecedents_text", "consequents_text"])
    else:
        itemsets["items"] = itemsets["itemsets"].map(lambda x: " + ".join(sorted(map(str, x))))
        if engine == "mlxtend":
            rules = association_rules(itemsets, metric="confidence", min_threshold=used_confidence) if len(itemsets) else pd.DataFrame()
        else:
            rules = _rules_fallback(itemsets, used_confidence)
        if rules.empty and len(itemsets):
            for attempt in range(3):
                if used_confidence <= 0.05: break
                used_confidence = round(used_confidence / 2, 4)
                notes.append(f"Confidence diturunkan otomatis ke {used_confidence} karena tidak ada rule pada ambang sebelumnya.")
                rules = association_rules(itemsets, metric="confidence", min_threshold=used_confidence) if engine == "mlxtend" else _rules_fallback(itemsets, used_confidence)
                if not rules.empty: break
        if rules.empty and len(itemsets):
            max_len_found = int(itemsets["itemsets"].map(len).max())
            if max_len_found < 2:
                notes.append("Tidak ada association rule yang bisa dibentuk: semua frequent itemset bertingkat 1 (satu produk). Turunkan minimum support agar itemset gabungan (≥2 produk) ditemukan.")
            else:
                notes.append("Itemset gabungan ditemukan, tetapi belum ada yang mencapai confidence minimum. Turunkan minimum confidence untuk melihat kandidat aturan.")
        if not rules.empty:
            rules["antecedents_text"] = rules["antecedents"].map(lambda x: " + ".join(sorted(map(str, x))))
            rules["consequents_text"] = rules["consequents"].map(lambda x: " + ".join(sorted(map(str, x))))
            rules = rules.sort_values(["lift","confidence","support"], ascending=False)
        w["rules"] = rules
        w["itemsets"] = itemsets.sort_values("support", ascending=False)
    w["status"]["apriori"] = "success"
    w["apriori_meta"] = {"min_support": used_support, "requested_support": min_support,
        "min_confidence": used_confidence, "requested_confidence": min_confidence,
        "transactions": int(matrix.shape[0]), "products": int(matrix.shape[1]),
        "engine": engine, "notes": notes,
        "support_absolute": int(round(used_support * matrix.shape[0])),
        "confidence_pct": round(used_confidence * 100, 1)}
    charts = dict(w.get("charts") or {})
    charts["top_itemsets"] = chartkit.top_itemsets_chart(w["itemsets"])
    charts["rules_scatter"] = chartkit.rules_scatter(w["rules"])
    w["charts"] = {k: v for k, v in charts.items() if v}
    w["recommendations"] = build_recommendations(w)
    return {"itemsets": len(w["itemsets"]), "rules": len(w["rules"]), "meta": w["apriori_meta"]}

def run_kmeans(w):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    df = w.get("clean")
    cols = w.get("columns", {})
    tx, cust, date, total = cols.get("transaction"), cols.get("customer"), cols.get("date"), cols.get("total")
    if df is None or not cust or not tx or not total:
        raise ValueError("K-Means RFM memerlukan kolom Customer_ID, Transaction_ID, dan Line_Total (atau kolom yang dipetakan).")
    tmp = df.dropna(subset=[cust, tx]).copy()
    if tmp.empty: raise ValueError("Tidak ada pelanggan dan transaksi valid untuk RFM.")
    ref_date = (tmp[date].max().normalize() + pd.Timedelta(days=1)) if date and tmp[date].notna().any() else pd.Timestamp.today().normalize()
    tmp["_value"] = pd.to_numeric(tmp[total], errors="coerce").fillna(0)
    rfm = tmp.groupby(cust).agg(
        last_purchase=(date, "max") if date else (tx, "last"),
        Frequency=(tx, "nunique"),
        Monetary=("_value", "sum")
    ).reset_index().rename(columns={cust:"Customer_ID"})
    if date:
        rfm["Recency"] = (ref_date - pd.to_datetime(rfm["last_purchase"], errors="coerce")).dt.days
    else:
        rfm["Recency"] = np.nan
    rfm = rfm.drop(columns=["last_purchase"])
    features = ["Frequency","Monetary"] + (["Recency"] if rfm["Recency"].notna().any() else [])
    rfm = rfm.replace([np.inf,-np.inf], np.nan).dropna(subset=features)
    if len(rfm) < 3: raise ValueError("Jumlah pelanggan valid terlalu sedikit untuk segmentasi.")
    # Log transform nonnegative skewed variables; shift monetary if negative values exist.
    model_df = rfm[features].astype(float).copy()
    for c in features:
        if (model_df[c] >= 0).all():
            model_df[c] = np.log1p(model_df[c])
    scaler = StandardScaler()
    X = scaler.fit_transform(model_df)
    max_k = min(8, len(rfm)-1)
    candidates = list(range(2, max_k+1))
    scores = []
    for k in candidates:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sil = float(silhouette_score(X, labels)) if len(set(labels)) > 1 and len(rfm) > k else None
        scores.append({"k":k,"inertia":float(km.inertia_),"silhouette":sil})
    # Select highest silhouette; fall back to elbow-compatible k=3 if possible.
    valid_scores = [s for s in scores if s["silhouette"] is not None]
    chosen = max(valid_scores, key=lambda x:x["silhouette"])["k"] if valid_scores else min(3, max_k)
    model = KMeans(n_clusters=chosen, random_state=42, n_init=10)
    rfm["Cluster"] = model.fit_predict(X)
    profile = rfm.groupby("Cluster").agg(Customer_Count=("Customer_ID","count"),
        Recency_Mean=("Recency","mean"), Frequency_Mean=("Frequency","mean"),
        Monetary_Mean=("Monetary","mean"), Recency_Median=("Recency","median"),
        Frequency_Median=("Frequency","median"), Monetary_Median=("Monetary","median")).reset_index()
    # Chart.js-ready series for the frontend (top monetary per cluster + RFM means).
    chart_clusters = [{"cluster": int(r["Cluster"]), "count": int(r["Customer_Count"]),
        "recency": round(float(r["Recency_Mean"]), 2), "frequency": round(float(r["Frequency_Mean"]), 2),
        "monetary": round(float(r["Monetary_Mean"]), 2),
        "monetary_median": round(float(r["Monetary_Median"]), 2)}
        for _, r in profile.iterrows()]
    w["rfm"], w["cluster_profile"], w["clustered"] = rfm, profile, rfm.copy()
    w["kmeans_meta"] = {"reference_date":str(ref_date.date()),"features":features,"scaling":"StandardScaler",
        "log_transform":"log1p pada fitur nonnegatif","candidate_scores":scores,"chosen_k":chosen,
        "selection_note":"k dipilih dari silhouette tertinggi di antara kandidat yang valid; tetap perlu interpretasi bisnis.",
        "clusters": chart_clusters, "total_customers": int(len(rfm))}
    charts = dict(w.get("charts") or {})
    charts["elbow_silhouette"] = chartkit.elbow_silhouette(scores)
    charts["cluster_scatter"] = chartkit.cluster_scatter_2d(rfm, tuple(features))
    charts["cluster_boxplot"] = chartkit.cluster_boxplot(rfm, tuple(features))
    w["charts"] = {k: v for k, v in charts.items() if v}
    w["status"]["kmeans"] = "success"
    w["recommendations"] = build_recommendations(w)
    return {"customers":len(rfm),"clusters":chosen,"profile":clean_json_records(profile,20),"meta":w["kmeans_meta"]}

@app.route("/")
def cover():
    return render_template("cover.html")

@app.route("/dashboard")
def dashboard():
    w = workspace()
    return render_template("index.html", nav=NAV, active="overview", has_data=w.get("raw") is not None,
                           source_name=w.get("source_name"), results=w.get("results"), status=w.get("status",{}))

@app.post("/start")
def start_system():
    name = (request.form.get("analyst") or "").strip()[:60]
    session["analyst"] = name or "Analis"
    return redirect(url_for("dashboard"))

@app.context_processor
def inject_analyst():
    return {"analyst": session.get("analyst")}

@app.route("/page/<page>")
def page(page):
    valid = {n[0] for n in NAV}
    if page not in valid: return "Halaman tidak ditemukan", 404
    w = workspace()
    rules = w.get("rules")
    itemsets = w.get("itemsets")
    kmeans_meta = w.get("kmeans_meta")
    # JSON-safe payloads for Chart.js on module pages.
    module_data = {"page": page}
    if page == "apriori":
        module_data["itemsets"] = clean_json_records(itemsets, 15) if itemsets is not None and not itemsets.empty else []
        if rules is not None and not rules.empty:
            module_data["rules"] = [{"antecedents": r.get("antecedents_text"), "consequents": r.get("consequents_text"),
                "support": json_safe(r.get("support")), "confidence": json_safe(r.get("confidence")),
                "lift": json_safe(r.get("lift"))} for _, r in rules.head(15).iterrows()]
        else:
            module_data["rules"] = []
        module_data["meta"] = w.get("apriori_meta")
        module_data["ready"] = w.get("apriori_ready")
    if page == "segmentation" and kmeans_meta:
        module_data["kmeans"] = {k: v for k, v in kmeans_meta.items() if k != "selection_note"}
        module_data["profiles"] = clean_json_records(w.get("cluster_profile"), 20)
        if w.get("rfm") is not None and not w["rfm"].empty:
            sample = w["rfm"]
            if len(sample) > 800:
                sample = sample.sample(800, random_state=42)
            module_data["scatter"] = [{"x": json_safe(r["Frequency"]), "y": json_safe(r["Monetary"]),
                "cluster": int(r["Cluster"])} for _, r in sample.iterrows()]
    if page == "recommendations":
        module_data["recommendations"] = w.get("recommendations", [])
        module_data["rules"] = (module_data.get("rules") or (
            [{"antecedents": r.get("antecedents_text"), "consequents": r.get("consequents_text"),
              "support": json_safe(r.get("support")), "confidence": json_safe(r.get("confidence")),
              "lift": json_safe(r.get("lift"))} for _, r in rules.head(8).iterrows()]
            if rules is not None and not rules.empty else []))
        module_data["clusters"] = kmeans_meta.get("clusters") if kmeans_meta else None
        module_data["results"] = {"top_products": (w.get("results") or {}).get("top_products", []),
                                  "category_sales": (w.get("results") or {}).get("category_sales", [])}
    if page == "preprocessing" and w.get("numeric_summary"):
        module_data["numeric_summary"] = w["numeric_summary"]
    return render_template("page.html", nav=NAV, active=page, page=page, has_data=w.get("raw") is not None,
        source_name=w.get("source_name"), results=w.get("results"), status=w.get("status",{}),
        profile=w.get("profiling"), quality=w.get("quality"), log=w.get("log",[]),
        recommendations=w.get("recommendations",[]), rules=w.get("rules"), itemsets=w.get("itemsets"),
        rfm=w.get("rfm"), cluster_profile=w.get("cluster_profile"), kmeans_meta=w.get("kmeans_meta"),
        charts=w.get("charts",{}), apriori_ready=w.get("apriori_ready"), apriori_meta=w.get("apriori_meta"),
        validation_audit=w.get("validation_audit"), numeric_summary=w.get("numeric_summary",[]),
        module_data=module_data)

@app.post("/api/upload")
def upload():
    w = workspace()
    f = request.files.get("dataset")
    if not f or not f.filename: return jsonify(error="Pilih file Excel atau CSV terlebih dahulu."), 400
    if not allowed(f.filename): return jsonify(error="Format tidak didukung. Gunakan .xlsx, .xls, atau .csv."), 400
    try:
        df, sheets, selected = read_uploaded(f)
        if df.empty or len(df.columns) == 0: return jsonify(error="Dataset kosong atau tidak memiliki kolom."), 400
        # Clear previous workspace state before accepting new dataset.
        w.clear()
        w.update({"source_name":secure_filename(f.filename), "raw":df.copy(deep=True), "sheets":sheets,
                  "active_sheet":selected, "created_at":datetime.now().isoformat(timespec="seconds"),
                  "status":{"upload":"success","eda":"pending","preprocessing":"pending","validation":"pending",
                            "apriori":"not_run","kmeans":"not_run","bi":"pending","recommendations":"pending"}})
        return jsonify(message="Dataset berhasil dibaca.", source_name=w["source_name"],
            rows=len(df), columns=len(df.columns), sheets=sheets, active_sheet=selected,
            column_names=[str(c) for c in df.columns], preview=clean_json_records(df,10))
    except Exception as e:
        return jsonify(error=f"Gagal membaca file: {e}"), 400

@app.post("/api/analyze")
def analyze():
    w = workspace()
    if w.get("raw") is None: return jsonify(error="Upload dataset terlebih dahulu."), 400
    try:
        results = calculate_analysis(w)
        return jsonify(message="EDA, preprocessing, validasi awal, dan KPI berhasil dihitung.", results=results,
                       status=w["status"], preview=clean_json_records(w["clean"],10))
    except Exception as e:
        return jsonify(error=f"Analisis gagal: {e}"), 500

@app.post("/api/apriori")
def apriori_api():
    w = workspace()
    if w.get("clean") is None: return jsonify(error="Jalankan analisis awal dan preprocessing terlebih dahulu."), 400
    try:
        support = float(request.form.get("min_support", 0.02))
        confidence = float(request.form.get("min_confidence", 0.3))
        return jsonify(run_apriori(w, min_support=support, min_confidence=confidence))
    except Exception as e: return jsonify(error=str(e)), 400

@app.get("/api/apriori/ready")
def apriori_ready_api():
    w = workspace()
    if w.get("clean") is None:
        return jsonify(ready=False, message="Jalankan analisis awal dan preprocessing terlebih dahulu."), 200
    return jsonify(ready=True, numbers=w.get("apriori_ready"), meta=w.get("apriori_meta"))

@app.post("/api/kmeans")
def kmeans_api():
    w = workspace()
    if w.get("clean") is None: return jsonify(error="Jalankan analisis awal dan preprocessing terlebih dahulu."), 400
    try: return jsonify(run_kmeans(w))
    except Exception as e: return jsonify(error=str(e)), 400

@app.post("/api/reset")
def reset():
    sid = session.get("sid")
    if sid: WORKSPACES.pop(sid, None)
    session.pop("sid", None)
    return jsonify(message="Workspace aktif direset. File sumber dan ekspor tersimpan tidak dihapus.")

@app.get("/api/state")
def state():
    w = workspace()
    return jsonify(has_data=w.get("raw") is not None, source_name=w.get("source_name"),
        results=w.get("results"), status=w.get("status",{}), sheets=w.get("sheets",[]),
        active_sheet=w.get("active_sheet"))

@app.get("/export/excel")
def export_excel():
    w = workspace()
    if w.get("raw") is None: return "Belum ada dataset.", 400
    if w.get("clean") is None: calculate_analysis(w)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame([{"Metrik":"Sumber","Nilai":w["source_name"]},
                      {"Metrik":"Waktu analisis","Nilai":w.get("results",{}).get("generated_at")},
                      {"Metrik":"Baris mentah","Nilai":len(w["raw"])},
                      {"Metrik":"Baris bersih","Nilai":len(w["clean"])}]).to_excel(writer,index=False,sheet_name="Executive_Summary")
        w["profiling"] and pd.DataFrame(w["profiling"]["column_profile"]).to_excel(writer,index=False,sheet_name="Data_Profiling")
        pd.DataFrame(w["quality"]["column_profile"]).to_excel(writer,index=False,sheet_name="Data_Quality")
        pd.DataFrame(w["log"]).to_excel(writer,index=False,sheet_name="Preprocessing_Log")
        w["clean"].to_excel(writer,index=False,sheet_name="Data_Clean")
        if w.get("itemsets") is not None and not w["itemsets"].empty:
            item = w["itemsets"].copy()
            if "itemsets" in item: item["itemsets"] = item["itemsets"].map(lambda x:", ".join(map(str,x)) if isinstance(x,frozenset) else str(x))
            item.to_excel(writer,index=False,sheet_name="Frequent_Itemsets")
        if w.get("rules") is not None and not w["rules"].empty:
            rules = w["rules"].copy()
            for c in ["antecedents","consequents"]:
                if c in rules: rules[c] = rules[c].map(lambda x:", ".join(map(str,x)) if isinstance(x,frozenset) else str(x))
            rules.to_excel(writer,index=False,sheet_name="Association_Rules")
        if w.get("rfm") is not None: w["rfm"].to_excel(writer,index=False,sheet_name="RFM_Analysis")
        if w.get("cluster_profile") is not None: w["cluster_profile"].to_excel(writer,index=False,sheet_name="Cluster_Profile")
        if w.get("recommendations"): pd.DataFrame(w["recommendations"]).to_excel(writer,index=False,sheet_name="Recommendations")
        if w.get("kmeans_meta"): pd.DataFrame(w["kmeans_meta"]["candidate_scores"]).to_excel(writer,index=False,sheet_name="KMeans_Evaluation")
        # Format workbook consistently.
        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                cell.fill = __import__("openpyxl").styles.PatternFill("solid", fgColor="172554")
                cell.font = __import__("openpyxl").styles.Font(color="FFFFFF", bold=True)
            for col_cells in ws.columns:
                letter = col_cells[0].column_letter
                width = min(42, max(12, max(len(str(c.value or "")) for c in col_cells[:200]) + 2))
                ws.column_dimensions[letter].width = width
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="Retail_Intelligence_Report.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

@app.get("/export/pdf")
def export_pdf():
    w = workspace()
    if w.get("raw") is None: return "Belum ada dataset.", 400
    if w.get("clean") is None: calculate_analysis(w)
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer,pagesize=A4,rightMargin=1.6*cm,leftMargin=1.6*cm,topMargin=1.6*cm,bottomMargin=1.6*cm)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="NavyTitle",parent=styles["Title"],textColor=colors.HexColor("#172554"),fontSize=24,leading=28))
    story=[Paragraph("RETAIL INTELLIGENCE",styles["NavyTitle"]),
        Paragraph("Business Intelligence & Data Mining Report",styles["Heading2"]),
        Spacer(1,0.5*cm),Paragraph(f"Sumber dataset: {w['source_name']}",styles["BodyText"]),
        Paragraph(f"Waktu analisis: {w['results'].get('generated_at')}",styles["BodyText"]),Spacer(1,0.4*cm),
        Paragraph("Ringkasan Eksekutif",styles["Heading1"])]
    r=w["results"]
    rows=[["Metrik","Hasil"],["Baris data mentah",str(r["raw_rows"])],["Baris data bersih",str(r["clean_rows"])],
          ["Kolom",str(r["clean_columns"])],["Transaksi unik",str(r.get("transactions") or "Tidak tersedia")],
          ["Pelanggan unik",str(r.get("customers") or "Tidak tersedia")],["Penjualan sementara tervalidasi",money(r.get("net_sales"))]]
    t=Table(rows,colWidths=[8*cm,8*cm],repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#172554")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
      ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#CBD5E1")),("PADDING",(0,0),(-1,-1),7)]))
    story += [t,Spacer(1,0.5*cm),Paragraph("Log Preprocessing",styles["Heading1"])]
    logrows=[["Tahap","Tindakan"]]+[[str(x["step"]),str(x["action"])[:180]] for x in w["log"]]
    lt=Table(logrows,colWidths=[5*cm,11*cm],repeatRows=1)
    lt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#0E7490")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
      ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CBD5E1")),("VALIGN",(0,0),(-1,-1),"TOP"),("PADDING",(0,0),(-1,-1),5)]))
    story += [lt,Spacer(1,0.4*cm)]
    if w.get("rules") is not None and not w["rules"].empty:
        story.append(Paragraph("Association Rules (Top 10)",styles["Heading1"]))
        rr=[["Antecedent","Consequent","Support","Confidence","Lift"]]
        for _,row in w["rules"].head(10).iterrows():
            rr.append([str(row.get("antecedents_text",""))[:45],str(row.get("consequents_text",""))[:45],
                f"{row.get('support',0):.3f}",f"{row.get('confidence',0):.3f}",f"{row.get('lift',0):.3f}"])
        rt=Table(rr,colWidths=[4*cm,4*cm,2*cm,2.5*cm,2*cm],repeatRows=1)
        rt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#172554")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
          ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CBD5E1")),("FONTSIZE",(0,0),(-1,-1),7),("PADDING",(0,0),(-1,-1),4)]))
        story.append(rt)
    if w.get("cluster_profile") is not None and not w["cluster_profile"].empty:
        story += [Spacer(1,0.4*cm),Paragraph("Profil Cluster K-Means",styles["Heading1"])]
        cp=w["cluster_profile"].round(2)
        story.append(Table([list(cp.columns)]+cp.values.tolist(),repeatRows=1,style=TableStyle([
          ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#0F766E")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
          ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#CBD5E1")),("FONTSIZE",(0,0),(-1,-1),7)])))
    story += [Spacer(1,0.5*cm),Paragraph("Catatan Metodologi & Keterbatasan",styles["Heading1"]),
      Paragraph("Laporan ini dibuat dari dataset yang diunggah pada sesi aktif. Nilai penjualan bersih merupakan agregasi sementara setelah filter validitas numerik yang tersedia; hasil harus ditafsirkan bersama log preprocessing. Outlier tidak otomatis dihapus. Hasil Apriori dan K-Means hanya ditampilkan jika algoritma berhasil dijalankan.",styles["BodyText"])]
    doc.build(story)
    buffer.seek(0)
    return send_file(buffer,as_attachment=True,download_name="Retail_Intelligence_Report.pdf",mimetype="application/pdf")

@app.errorhandler(413)
def too_large(_): return jsonify(error="File terlalu besar. Batas unggahan adalah 100 MB."), 413

if __name__ == "__main__":
    print("Retail Intelligence berjalan di http://127.0.0.1:5000")
    app.run(debug=True)
