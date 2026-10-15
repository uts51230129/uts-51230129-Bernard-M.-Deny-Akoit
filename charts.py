"""Matplotlib chart helpers rendered server-side as base64 PNG images.

All figures use a dark theme that matches the CSS design system so the images
blend into the dashboard panels.
"""
from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BG = "#0b1220"
PANEL = "#131f33"
TEXT = "#c2cfdf"
MUTED = "#8fa1b9"
GRID = "#26364f"
PALETTE = ["#3b82f6", "#06b6d4", "#14b8a6", "#7c3aed", "#f59e0b", "#22c55e", "#ef4444", "#a78bfa"]


def _new_fig(width, height):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor(PANEL)
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, alpha=0.45, linewidth=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def _fig_to_b64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    buffer.seek(0)
    return "data:image/png;base64," + base64.b64encode(buffer.read()).decode("ascii")


def _fmt_id(value):
    """Compact Indonesian number label for chart annotations."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(value) >= 1000:
        return f"{value:,.0f}".replace(",", ".")
    if value == int(value):
        return str(int(value))
    return f"{value:.2f}".replace(".", ",")


def numeric_boxplot(df: pd.DataFrame, columns=None, log_scale=True, title=""):
    """Boxplot for numeric columns; returns base64 PNG or None."""
    if df is None or df.empty:
        return None
    numeric = df.select_dtypes(include=np.number)
    cols = [c for c in (columns or numeric.columns) if c in numeric.columns]
    cols = [c for c in cols if numeric[c].notna().any()][:6]
    if not cols:
        return None
    n = len(cols)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    fig.patch.set_facecolor(PANEL)
    for ax, col in zip(axes, cols):
        ax.set_facecolor(PANEL)
        values = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
        if values.size == 0:
            ax.set_visible(False)
            continue
        show_fliers = values.size <= 4000
        if log_scale and (values > 0).all() and (np.nanpercentile(values, 75) / max(np.nanpercentile(values, 25), 1e-9)) > 50:
            plot_vals = np.log10(np.clip(values, 1e-6, None))
            bp = ax.boxplot(plot_vals, patch_artist=True, widths=0.5, showfliers=show_fliers)
            ax.set_ylabel("log10 skala", color=MUTED, fontsize=7)
            q1, q3 = np.nanpercentile(values, [25, 75])
            iqr = q3 - q1
            outliers = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())
        else:
            bp = ax.boxplot(values, patch_artist=True, widths=0.5, showfliers=show_fliers)
            q1, q3 = np.nanpercentile(values, [25, 75])
            iqr = q3 - q1
            outliers = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())
        for patch in bp["boxes"]:
            patch.set_facecolor("#3b82f6")
            patch.set_alpha(0.45)
            patch.set_edgecolor("#60a5fa")
        for median in bp["medians"]:
            median.set_color("#22d3ee")
            median.set_linewidth(1.6)
        for whisker in bp["whiskers"]:
            whisker.set_color("#64748b")
        for cap in bp["caps"]:
            cap.set_color("#64748b")
        ax.set_title(str(col), color=TEXT, fontsize=9, pad=8)
        ax.text(0.5, -0.16, f"outlier IQR: {_fmt_id(outliers)}", transform=ax.transAxes,
                ha="center", color=MUTED, fontsize=7)
    for ax in axes[n:]:
        ax.set_visible(False)
    if title:
        fig.suptitle(title, color=TEXT, fontsize=10, y=1.02)
    fig.tight_layout()
    return _fig_to_b64(fig)


def missing_before_after(profile, quality, top=12):
    """Grouped bar chart: missing values per column before vs after cleaning."""
    if not profile or not quality:
        return None
    before = {c["column"]: c.get("missing", 0) for c in profile.get("column_profile", [])}
    after = {c["column"]: c.get("missing", 0) for c in quality.get("column_profile", [])}
    names = [c for c in before if before.get(c, 0) or after.get(c, 0)]
    names = sorted(names, key=lambda c: -(before.get(c, 0) + after.get(c, 0)))[:top]
    if not names:
        return None
    fig, ax = _new_fig(7.2, 3.2)
    x = np.arange(len(names))
    width = 0.38
    b_vals = [before.get(c, 0) for c in names]
    a_vals = [after.get(c, 0) for c in names]
    ax.bar(x - width / 2, b_vals, width, label="Sebelum", color="#f59e0b", alpha=0.85)
    ax.bar(x + width / 2, a_vals, width, label="Sesudah", color="#14b8a6", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=7)
    ax.set_title("Missing values per kolom: sebelum vs sesudah", color=TEXT, fontsize=10, pad=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT)
    ax.tick_params(axis="y", labelsize=7)
    return _fig_to_b64(fig)


def rows_flow(log):
    """Line/bar chart tracking row counts through preprocessing steps."""
    if not log:
        return None
    steps = [str(l.get("step", f"Tahap {i+1}"))[:34] for i, l in enumerate(log)]
    before = [float(l.get("before") or 0) for l in log]
    after = [float(l.get("after") or 0) for l in log]
    fig, ax = _new_fig(7.2, 3.2)
    x = np.arange(len(steps))
    ax.plot(x, before, marker="o", color="#3b82f6", linewidth=1.6, label="Baris sebelum")
    ax.plot(x, after, marker="s", color="#14b8a6", linewidth=1.6, label="Baris sesudah")
    ax.fill_between(x, after, before, color="#3b82f6", alpha=0.10)
    for i, (b, a) in enumerate(zip(before, after)):
        ax.annotate(_fmt_id(b), (i, b), textcoords="offset points", xytext=(0, 6),
                    ha="center", color="#93c5fd", fontsize=6.5)
        ax.annotate(_fmt_id(a), (i, a), textcoords="offset points", xytext=(0, -11),
                    ha="center", color="#5eead4", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(steps, rotation=18, ha="right", fontsize=6.8)
    ax.set_title("Alur jumlah baris tiap tahap preprocessing", color=TEXT, fontsize=10, pad=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT)
    ax.tick_params(axis="y", labelsize=7)
    return _fig_to_b64(fig)


def audit_severity_chart(audit_rows):
    """Horizontal bar of audit findings grouped by severity."""
    if not audit_rows:
        return None
    order = ["Tinggi", "Sedang", "Rendah", "Info"]
    counts = {s: 0 for s in order}
    for row in audit_rows:
        severity = str(row.get("severity", "Info"))
        counts[severity] = counts.get(severity, 0) + int(row.get("count", 0) or 0)
    labels = [s for s in order if counts.get(s, 0) > 0]
    if not labels:
        return None
    values = [counts[s] for s in labels]
    colors = {"Tinggi": "#ef4444", "Sedang": "#f59e0b", "Rendah": "#3b82f6", "Info": "#64748b"}
    fig, ax = _new_fig(6.4, 2.6)
    bars = ax.barh(labels[::-1], values[::-1], color=[colors[s] for s in labels[::-1]], alpha=0.85)
    for bar, value in zip(bars, values[::-1]):
        ax.text(bar.get_width() + max(values) * 0.015, bar.get_y() + bar.get_height() / 2,
                _fmt_id(value), va="center", color=TEXT, fontsize=8)
    ax.set_title("Temuan audit berdasarkan tingkat keparahan", color=TEXT, fontsize=10, pad=10)
    ax.xaxis.grid(True, color=GRID, alpha=0.45, linewidth=0.6)
    ax.yaxis.grid(False)
    ax.tick_params(axis="x", labelsize=7)
    return _fig_to_b64(fig)


def rules_scatter(rules_df, max_points=300):
    """Scatter support vs confidence, point size ~ lift, color by lift."""
    if rules_df is None or rules_df.empty:
        return None
    df = rules_df.head(max_points)
    fig, ax = _new_fig(7.2, 3.4)
    sizes = 30 + df["lift"].astype(float).to_numpy() * 55
    scatter = ax.scatter(df["support"], df["confidence"], c=df["lift"], cmap="viridis",
                         s=np.clip(sizes, 25, 260), alpha=0.8, edgecolors="#0b1220", linewidths=0.4)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Lift", color=MUTED, fontsize=8)
    cbar.ax.tick_params(colors=MUTED, labelsize=7)
    ax.set_xlabel("Support", color=MUTED, fontsize=8)
    ax.set_ylabel("Confidence", color=MUTED, fontsize=8)
    ax.set_title("Peta association rules (support vs confidence, warna = lift)", color=TEXT, fontsize=10, pad=10)
    return _fig_to_b64(fig)


def top_itemsets_chart(itemsets_df, top=12):
    """Horizontal bar of top frequent itemsets by support."""
    if itemsets_df is None or itemsets_df.empty or "items" not in itemsets_df:
        return None
    df = itemsets_df.head(top).iloc[::-1]
    fig, ax = _new_fig(7.0, 3.4)
    bars = ax.barh(df["items"].astype(str), df["support"].astype(float), color="#3b82f6", alpha=0.85)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.1%}".replace(".", ","))
    for bar, value in zip(bars, df["support"].astype(float)):
        ax.text(bar.get_width() + max(df["support"].astype(float)) * 0.012,
                bar.get_y() + bar.get_height() / 2, f"{value:.1%}".replace(".", ","),
                va="center", color=TEXT, fontsize=7.5)
    ax.set_title("Itemset paling sering muncul (top support)", color=TEXT, fontsize=10, pad=10)
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", labelsize=7)
    ax.xaxis.grid(True, color=GRID, alpha=0.45, linewidth=0.6)
    ax.yaxis.grid(False)
    return _fig_to_b64(fig)


def elbow_silhouette(scores):
    """Combined inertia line + silhouette bars for candidate k."""
    if not scores:
        return None
    ks = [s["k"] for s in scores]
    inertia = [s["inertia"] for s in scores]
    silhouette = [s.get("silhouette") for s in scores]
    fig, ax1 = _new_fig(7.0, 3.0)
    ax1.plot(ks, inertia, marker="o", color="#3b82f6", linewidth=1.6, label="Inertia")
    ax1.set_xlabel("Jumlah cluster (k)", color=MUTED, fontsize=8)
    ax1.set_ylabel("Inertia", color=MUTED, fontsize=8)
    ax1.tick_params(axis="x", labelsize=8)
    ax2 = ax1.twinx()
    ax2.set_facecolor(PANEL)
    valid = [(k, s) for k, s in zip(ks, silhouette) if s is not None]
    if valid:
        ax2.bar([k for k, _ in valid], [s for _, s in valid], color="#14b8a6", alpha=0.35, width=0.55, label="Silhouette")
        ax2.set_ylabel("Silhouette", color=MUTED, fontsize=8)
    for spine in ax2.spines.values():
        spine.set_color(GRID)
    ax2.tick_params(colors=MUTED, labelsize=7)
    lines = ax1.get_lines() + ax2.patches[:1]
    labels = ["Inertia (elbow)", "Silhouette score"]
    ax1.legend(lines, labels, frameon=False, fontsize=8, labelcolor=TEXT, loc="upper right")
    ax1.set_title("Elbow & silhouette: evaluasi kandidat k", color=TEXT, fontsize=10, pad=10)
    ax1.set_xticks(ks)
    return _fig_to_b64(fig)


def cluster_scatter_2d(rfm, features=("Recency", "Frequency", "Monetary")):
    """2D projection (first two PCA-ish scaled features) colored by cluster."""
    if rfm is None or rfm.empty or "Cluster" not in rfm:
        return None
    cols = [c for c in features if c in rfm.columns]
    if len(cols) < 2:
        return None
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    data = rfm[cols].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 3:
        return None
    transformed = StandardScaler().fit_transform(np.log1p(np.clip(data.to_numpy(), 0, None)))
    coords = PCA(n_components=2, random_state=42).fit_transform(transformed)
    labels = rfm.loc[data.index, "Cluster"].to_numpy()
    fig, ax = _new_fig(7.0, 3.6)
    for cluster in sorted(set(labels.tolist())):
        mask = labels == cluster
        color = PALETTE[int(cluster) % len(PALETTE)]
        ax.scatter(coords[mask, 0], coords[mask, 1], s=16, alpha=0.72, color=color,
                   edgecolors="#0b1220", linewidths=0.25, label=f"Cluster {cluster}")
    ax.set_xlabel("Komponen 1 (PCA)", color=MUTED, fontsize=8)
    ax.set_ylabel("Komponen 2 (PCA)", color=MUTED, fontsize=8)
    ax.set_title("Visualisasi cluster pelanggan (proyeksi 2D fitur RFM)", color=TEXT, fontsize=10, pad=10)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=TEXT, markerscale=1.6)
    return _fig_to_b64(fig)


def cluster_boxplot(rfm, features=("Recency", "Frequency", "Monetary")):
    """Boxplot distribusi tiap fitur RFM per cluster."""
    if rfm is None or rfm.empty or "Cluster" not in rfm:
        return None
    cols = [c for c in features if c in rfm.columns]
    if not cols:
        return None
    clusters = sorted(rfm["Cluster"].dropna().unique().tolist())
    fig, axes = plt.subplots(1, len(cols), figsize=(3.2 * len(cols), 2.9))
    axes = np.atleast_1d(axes).ravel()
    fig.patch.set_facecolor(PANEL)
    for ax, col in zip(axes, cols):
        ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=7)
        ax.yaxis.grid(True, color=GRID, alpha=0.45, linewidth=0.6)
        ax.set_axisbelow(True)
        data = [pd.to_numeric(rfm.loc[rfm["Cluster"] == c, col], errors="coerce").dropna().to_numpy() for c in clusters]
        box_kwargs = dict(patch_artist=True, widths=0.55)
        try:
            bp = ax.boxplot([d if d.size else np.array([np.nan]) for d in data],
                            tick_labels=[f"C{c}" for c in clusters], **box_kwargs)
        except TypeError:
            bp = ax.boxplot([d if d.size else np.array([np.nan]) for d in data],
                            labels=[f"C{c}" for c in clusters], **box_kwargs)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(PALETTE[i % len(PALETTE)])
            patch.set_alpha(0.45)
            patch.set_edgecolor(PALETTE[i % len(PALETTE)])
        for median in bp["medians"]:
            median.set_color("#22d3ee")
            median.set_linewidth(1.5)
        for whisker in bp["whiskers"]:
            whisker.set_color("#64748b")
        for cap in bp["caps"]:
            cap.set_color("#64748b")
        ax.set_title(col, color=TEXT, fontsize=9, pad=8)
    fig.tight_layout()
    return _fig_to_b64(fig)


def distribution_chart(series, title="", bins=24, color="#3b82f6"):
    """Histogram + mean/median markers for a numeric series."""
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if values.size == 0:
        return None
    fig, ax = _new_fig(7.0, 2.9)
    ax.hist(values, bins=min(bins, max(6, int(np.sqrt(values.size)))), color=color, alpha=0.8, edgecolor="#0b1220")
    mean, median = float(np.mean(values)), float(np.median(values))
    ax.axvline(mean, color="#f59e0b", linewidth=1.4, linestyle="--", label=f"Mean {_fmt_id(mean)}")
    ax.axvline(median, color="#22d3ee", linewidth=1.4, linestyle="--", label=f"Median {_fmt_id(median)}")
    ax.set_title(title or "Distribusi", color=TEXT, fontsize=10, pad=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT)
    ax.tick_params(axis="x", labelsize=7)
    return _fig_to_b64(fig)
