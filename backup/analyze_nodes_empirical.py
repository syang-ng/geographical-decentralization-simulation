#!/usr/bin/env python3
"""
Geographical decentralization metrics & plots for Ethereum node IPs by date,
PLUS interactive world maps (overall, latest, animated, choropleth).

Input CSV format:
    date,number_ips,ips
    2024-12-18,2975,"['23.227.207.174','195.224.80.26','175.114.129.9', ...]"

Outputs under --outdir:
  - metrics_over_time.csv
  - Time-series figures: clusters, total_distance, avg_nnd, nni, unique_countries, country_entropy
  - Country bars (overall & final date)
  - all_metrics_panel.(pdf|png) when --panel
  - Map HTMLs when --maps: map_overall.html, map_latest.html, map_animated.html, choropleth_latest.html

Dependencies:
  numpy, pandas, matplotlib, requests, plotly (for maps)
  Your modules: distribution.py (SphericalSpace, init_distance_matrix)
                measure.py (cluster_matrix, total_distance, average_nearest_neighbor_distance, nearest_neighbor_index_spherical)
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

# spatial helpers from your codebase
from distribution import SphericalSpace, init_distance_matrix
from measure import (
    cluster_matrix,
    total_distance,
    average_nearest_neighbor_distance,
    nearest_neighbor_index_spherical,
)

# Try plotly for maps
try:
    import plotly.express as px
    import plotly.io as pio

    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

DEFAULT_OUTDIR = "figures"
DEFAULT_PANEL_SIZE = "12x10"
SPACE = SphericalSpace()


# ------------------------ helpers ------------------------
def latlon_to_point(lat_deg: float, lon_deg: float) -> Tuple[float, float, float]:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    x = math.cos(lat) * math.cos(lon)
    y = math.cos(lat) * math.sin(lon)
    z = math.sin(lat)
    return (x, y, z)


def safe_literal_list(s: str) -> List[str]:
    if isinstance(s, list):
        return s
    if not isinstance(s, str):
        return []
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (list, tuple)):
            return [str(x) for x in val]
    except Exception:
        pass
    s = s.strip().strip('"').strip("'").replace("[", "").replace("]", "")
    parts = [p.strip().strip("'").strip('"') for p in s.split(",")]
    return [p for p in parts if p]


def maybe_rolling(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr
    if window % 2 == 0:
        window += 1
    k = window // 2
    x = np.concatenate([np.full(k, np.nan), arr, np.full(k, np.nan)])
    out = np.empty_like(arr, dtype=float)
    for i in range(len(arr)):
        out[i] = np.nanmean(x[i : i + window])
    return out


def shannon_entropy(labels: Sequence[str]) -> float:
    if not labels:
        return 0.0
    counts = Counter(labels)
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    probs = [c / total for c in counts.values()]
    return float(-sum(p * math.log(p, 2) for p in probs if p > 0))


# ------------------------ geolocation ------------------------
@dataclass
class GeoRecord:
    lat: float
    lon: float
    country: str


class GeoResolver:
    """MaxMind (if provided) -> ip-api.com fallback with caching."""

    def __init__(
        self, geolite_path: Optional[Path], cache_path: Path, pause: float = 0.45
    ):
        self.cache_path = cache_path
        self.pause = pause
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._load_cache()
        self.reader = None
        self.use_maxmind = False
        if geolite_path:
            try:
                import geoip2.database  # type: ignore

                self.reader = geoip2.database.Reader(str(geolite_path))
                self.use_maxmind = True
            except Exception:
                self.reader = None
                self.use_maxmind = False

    def _load_cache(self):
        if self.cache_path.is_file():
            try:
                self.cache = json.loads(self.cache_path.read_text())
            except Exception:
                self.cache = {}

    def _save_cache(self):
        try:
            self.cache_path.write_text(json.dumps(self.cache))
        except Exception:
            pass

    def _mm_lookup(self, ip: str) -> Optional[GeoRecord]:
        if not self.reader:
            return None
        try:
            r = self.reader.city(ip)
            if (
                not r
                or not r.location
                or r.location.latitude is None
                or r.location.longitude is None
            ):
                return None
            lat = float(r.location.latitude)
            lon = float(r.location.longitude)
            country = (r.country.name or "Unknown") if r.country else "Unknown"
            return GeoRecord(lat, lon, country)
        except Exception:
            return None

    def _api_lookup(self, ip: str) -> Optional[GeoRecord]:
        try:
            url = f"http://ip-api.com/json/{ip}?fields=status,country,lat,lon,message"
            resp = requests.get(url, timeout=8)
            if resp.status_code != 200:
                return None
            js = resp.json()
            if js.get("status") != "success":
                return None
            lat = float(js.get("lat"))
            lon = float(js.get("lon"))
            country = str(js.get("country") or "Unknown")
            time.sleep(self.pause)
            return GeoRecord(lat, lon, country)
        except Exception:
            return None

    def resolve(self, ip: str) -> Optional[GeoRecord]:
        if ip in self.cache:
            c = self.cache[ip]
            try:
                return GeoRecord(float(c["lat"]), float(c["lon"]), str(c["country"]))
            except Exception:
                pass
        rec = self._mm_lookup(ip) if self.use_maxmind else None
        if rec is None:
            rec = self._api_lookup(ip)
        if rec is not None:
            self.cache[ip] = {"lat": rec.lat, "lon": rec.lon, "country": rec.country}
            self._save_cache()
        return rec


# ------------------------ metrics ------------------------
def safe_metrics_for_points(
    points: Sequence[Tuple[float, float, float]],
) -> Tuple[float, float, float, float]:
    if not points or len(points) <= 1:
        return 0.0, 0.0, 0.0, 0.0
    dm = init_distance_matrix(points, SPACE)
    c = float(cluster_matrix(dm))
    td = float(total_distance(dm))
    nnd = float(average_nearest_neighbor_distance(dm))
    nni = float(nearest_neighbor_index_spherical(dm, SPACE)[0])
    return c, td, nnd, nni


# ------------------------ plotting (matplotlib) ------------------------
def _fig_size(size_str: str, fallback: Tuple[float, float]) -> Tuple[float, float]:
    try:
        w, h = size_str.lower().split("x")
        return float(w), float(h)
    except Exception:
        return fallback


def save_fig(fig: plt.Figure, out: Path, fmt: str = "pdf", dpi: int = 300):
    out = out.with_suffix(f".{fmt}")
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    if fmt.lower() != "pdf":
        fig.savefig(out.with_suffix(".pdf"))
    print(f"✓ {out}")


def plot_ts(x, y, title, ylabel, outfile: Path, size="10x4", fmt="pdf", dpi=300):
    w, h = _fig_size(size, (10.0, 4.0))
    fig, ax = plt.subplots(figsize=(w, h))
    ax.plot(x, y, lw=1.6)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    save_fig(fig, outfile, fmt, dpi)
    plt.close(fig)


def bar_top(ax: plt.Axes, items: List[Tuple[str, float]], title: str, xlabel: str):
    if not items:
        ax.set_axis_off()
        ax.set_title(title + " (no data)")
        return
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    ax.bar(range(len(vals)), vals)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Nodes")


# ------------------------ plotly maps ------------------------
def save_plotly(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path))
    print(f"✓ {path}")


def make_maps(
    outdir: Path,
    per_date_points: Dict[pd.Timestamp, List[Tuple[float, float, str]]],
    per_date_counts_by_country: Dict[pd.Timestamp, Dict[str, int]],
    max_points: int,
):
    if not PLOTLY_OK:
        print(
            "! plotly not installed; skipping maps. Run `pip install plotly` if you want maps."
        )
        return
    # Build a long DataFrame for scatter map
    rows = []
    for d, pts in per_date_points.items():
        # downsample per date if needed
        if max_points and len(pts) > max_points:
            # simple uniform subsample
            step = max(1, len(pts) // max_points)
            pts = pts[::step]
        for lat, lon, country in pts:
            rows.append({"date": d, "lat": lat, "lon": lon, "country": country})
    if not rows:
        print("! No geolocated points for maps.")
        return
    df_pts = pd.DataFrame(rows).sort_values("date")
    df_pts["date_str"] = df_pts["date"].dt.strftime("%Y-%m-%d")

    # Overall scatter
    fig_overall = px.scatter_geo(
        df_pts,
        lat="lat",
        lon="lon",
        hover_name="country",
        title="All Geolocated Nodes (Overall)",
        opacity=0.6,
        size_max=4,
    )
    fig_overall.update_geos(projection_type="natural earth", showcountries=True)
    save_plotly(fig_overall, outdir / "map_overall.html")

    # Latest date scatter
    latest = df_pts["date"].max()
    df_latest = df_pts[df_pts["date"] == latest]
    fig_latest = px.scatter_geo(
        df_latest,
        lat="lat",
        lon="lon",
        hover_name="country",
        title=f"Nodes on {latest.date()}",
        opacity=0.7,
        size_max=6,
    )
    fig_latest.update_geos(projection_type="natural earth", showcountries=True)
    save_plotly(fig_latest, outdir / "map_latest.html")

    # Animated scatter over time
    fig_anim = px.scatter_geo(
        df_pts,
        lat="lat",
        lon="lon",
        hover_name="country",
        animation_frame="date_str",
        title="Nodes Over Time (Animated)",
        opacity=0.7,
        size_max=5,
    )
    fig_anim.update_geos(projection_type="natural earth", showcountries=True)
    fig_anim.update_layout(transition={"duration": 200})
    save_plotly(fig_anim, outdir / "map_animated.html")

    # Choropleth for latest date
    # Build country count frame for latest
    latest_counts = per_date_counts_by_country.get(latest, {})
    if latest_counts:
        df_choro = pd.DataFrame(
            [{"country": c, "count": n} for c, n in latest_counts.items()]
        )
        fig_choro = px.choropleth(
            df_choro,
            locations="country",
            locationmode="country names",
            color="count",
            color_continuous_scale="Blues",
            title=f"Node Count by Country on {latest.date()}",
        )
        save_plotly(fig_choro, outdir / "choropleth_latest.html")


# ------------------------ main pipeline ------------------------
def run(
    csv_path: Path,
    outdir: Path,
    geolite_path: Optional[Path],
    cache_path: Path,
    rolling: int,
    fmt: str,
    dpi: int,
    size_ts: str,
    panel: bool,
    panel_size: str,
    maps: bool,
    max_points: int,
):
    outdir.mkdir(parents=True, exist_ok=True)
    resolver = GeoResolver(geolite_path, cache_path)

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.sort_values("date")

    records: List[Dict[str, Any]] = []
    overall_country_counts: Counter[str] = Counter()
    final_date_countries: Counter[str] = Counter()
    # For maps
    per_date_points: Dict[pd.Timestamp, List[Tuple[float, float, str]]] = defaultdict(
        list
    )
    per_date_counts_by_country: Dict[pd.Timestamp, Dict[str, int]] = {}

    for _, row in df.iterrows():
        date = row["date"]
        ips = safe_literal_list(row.get("ips", ""))

        points_xyz: List[Tuple[float, float, float]] = []
        countries: List[str] = []
        latlons: List[Tuple[float, float, str]] = []

        for ip in ips:
            rec = resolver.resolve(ip)
            if rec is None:
                continue
            points_xyz.append(latlon_to_point(rec.lat, rec.lon))
            countries.append(rec.country)
            latlons.append((rec.lat, rec.lon, rec.country))

        overall_country_counts.update(countries)
        final_date_countries = Counter(countries)
        per_date_points[date] = latlons
        per_date_counts_by_country[date] = dict(Counter(countries))

        clusters, total_dist, avg_nnd, nni = safe_metrics_for_points(points_xyz)
        records.append(
            {
                "date": date,
                "num_ips_raw": int(row.get("number_ips", len(ips))),
                "num_ips_geolocated": len(points_xyz),
                "clusters": clusters,
                "total_distance": total_dist,
                "avg_nnd": avg_nnd,
                "nni": nni,
                "unique_countries": len(set(countries)),
                "country_entropy": shannon_entropy(countries),
            }
        )

    if not records:
        raise SystemExit("No metrics computed (no rows or geolocation failed).")

    out_df = pd.DataFrame(records).sort_values("date")
    if rolling and rolling > 1:
        for col in [
            "clusters",
            "total_distance",
            "avg_nnd",
            "nni",
            "unique_countries",
            "country_entropy",
        ]:
            out_df[col] = maybe_rolling(out_df[col].to_numpy(dtype=float), rolling)

    metrics_csv = outdir / "metrics_over_time.csv"
    out_df.to_csv(metrics_csv, index=False)
    print(f"✓ {metrics_csv}")

    # time-series plots
    x = out_df["date"].values
    plot_ts(
        x,
        out_df["clusters"].values,
        "# Clusters",
        "clusters",
        outdir / "clusters_over_time",
        size=size_ts,
        fmt=fmt,
        dpi=dpi,
    )
    plot_ts(
        x,
        out_df["total_distance"].values,
        "Total Distance",
        "distance",
        outdir / "total_distance_over_time",
        size=size_ts,
        fmt=fmt,
        dpi=dpi,
    )
    plot_ts(
        x,
        out_df["avg_nnd"].values,
        "Avg Nearest-Neighbor Distance",
        "distance",
        outdir / "avg_nnd_over_time",
        size=size_ts,
        fmt=fmt,
        dpi=dpi,
    )
    plot_ts(
        x,
        out_df["nni"].values,
        "Nearest Neighbor Index (NNI)",
        "nni",
        outdir / "nni_over_time",
        size=size_ts,
        fmt=fmt,
        dpi=dpi,
    )
    plot_ts(
        x,
        out_df["unique_countries"].values,
        "Unique Countries",
        "count",
        outdir / "unique_countries_over_time",
        size=size_ts,
        fmt=fmt,
        dpi=dpi,
    )
    plot_ts(
        x,
        out_df["country_entropy"].values,
        "Country Entropy (Shannon)",
        "bits",
        outdir / "country_entropy_over_time",
        size=size_ts,
        fmt=fmt,
        dpi=dpi,
    )

    # country bars
    top_overall = sorted(
        overall_country_counts.items(), key=lambda kv: kv[1], reverse=True
    )[:15]
    top_final = sorted(
        final_date_countries.items(), key=lambda kv: kv[1], reverse=True
    )[:15]
    w, h = _fig_size("12x4", (12.0, 4.0))
    fig, ax = plt.subplots(figsize=(w, h))
    bar_top(ax, top_overall, "Top Countries (Overall)", "Country")
    save_fig(fig, outdir / "countries_overall", fmt, dpi)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(w, h))
    bar_top(ax, top_final, "Top Countries (Final Date)", "Country")
    save_fig(fig, outdir / "countries_final", fmt, dpi)
    plt.close(fig)

    # panel
    if panel:
        cols = 3
        panels = [
            (
                "# Clusters",
                out_df["clusters"].to_numpy(dtype=float),
                "Date",
                "clusters",
            ),
            (
                "Total Distance",
                out_df["total_distance"].to_numpy(dtype=float),
                "Date",
                "distance",
            ),
            ("Avg NND", out_df["avg_nnd"].to_numpy(dtype=float), "Date", "distance"),
            ("NNI", out_df["nni"].to_numpy(dtype=float), "Date", "nni"),
            (
                "Unique Countries",
                out_df["unique_countries"].to_numpy(dtype=float),
                "Date",
                "count",
            ),
            (
                "Country Entropy",
                out_df["country_entropy"].to_numpy(dtype=float),
                "Date",
                "bits",
            ),
        ]
        rows = int(math.ceil(len(panels) / cols))
        pw, ph = _fig_size(panel_size, (12.0, 10.0))
        fig, axes = plt.subplots(rows, cols, figsize=(pw, ph))
        axes = np.atleast_2d(axes)
        for i, (title, series, xlabel, ylabel) in enumerate(panels):
            r, c = divmod(i, cols)
            ax = axes[r, c]
            ax.plot(out_df["date"].values, series, lw=1.2)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
        for j in range(len(panels), rows * cols):
            r, c = divmod(j, cols)
            axes[r, c].set_axis_off()
        fig.suptitle("Geographical Decentralization Metrics (by Date)", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        save_fig(fig, outdir / "all_metrics_panel", fmt, dpi)
        plt.close(fig)

    # maps
    if maps:
        make_maps(
            outdir, per_date_points, per_date_counts_by_country, max_points=max_points
        )


def parse_args():
    ap = argparse.ArgumentParser(
        description="Compute & plot geographical decentralization metrics by date from IP CSV, with world maps."
    )
    ap.add_argument(
        "--csv", required=True, help="Path to CSV with columns: date,number_ips,ips"
    )
    ap.add_argument(
        "--outdir", default=DEFAULT_OUTDIR, help="Output directory for figures and CSV"
    )
    ap.add_argument(
        "--geolite",
        default=None,
        help="Path to GeoLite2-City.mmdb (optional; if missing, uses ip-api.com)",
    )
    ap.add_argument(
        "--cache",
        default="ip_cache.json",
        help="Path to JSON cache for IP geolocations",
    )
    ap.add_argument(
        "--rolling",
        type=int,
        default=1,
        help="Centered rolling window (odd recommended) for smoothing",
    )
    ap.add_argument(
        "--fmt",
        default="pdf",
        choices=["pdf", "png"],
        help="Output figure format for static plots",
    )
    ap.add_argument("--dpi", type=int, default=300, help="Raster DPI (for PNG)")
    ap.add_argument("--size", default="10x4", help="Per-plot figure size, e.g., 12x4")
    ap.add_argument(
        "--panel",
        action="store_true",
        help="Also save a combined panel with all metrics",
    )
    ap.add_argument(
        "--panel-size", default=DEFAULT_PANEL_SIZE, help="Panel size, e.g., 12x10"
    )
    ap.add_argument(
        "--maps", action="store_true", help="Generate interactive map HTMLs via plotly"
    )
    ap.add_argument(
        "--max-points",
        type=int,
        default=6000,
        help="Max points per date for map plots (downsample to keep HTML light)",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        csv_path=Path(args.csv),
        outdir=Path(args.outdir),
        geolite_path=Path(args.geolite) if args.geolite else None,
        cache_path=Path(args.cache),
        rolling=max(1, args.rolling),
        fmt=args.fmt,
        dpi=args.dpi,
        size_ts=args.size,
        panel=bool(args.panel),
        panel_size=args.panel_size,
        maps=bool(args.maps),
        max_points=max(0, args.max_points),
    )
