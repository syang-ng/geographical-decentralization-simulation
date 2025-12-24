# analysis_utils.py
from __future__ import annotations

import argparse
import json
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt

from distribution import SphericalSpace, init_distance_matrix
from measure import (
    cluster_matrix,
    total_distance,
    average_nearest_neighbor_distance,
    nearest_neighbor_index_spherical,
)

import seaborn as sns
sns.set_style("whitegrid")

SPACE = SphericalSpace()


data_path = Path("data")

region_maps = {
    'australia': 'Oceania',
    'asia': 'Asia',
    'africa': 'Africa',
    'europe': 'Europe',
    'northamerica': 'North America',
    'southamerica': 'South America',
    'us': 'North America',
    'me': 'Middle East',
    "Asia": "Asia",
    "Europe": "Europe",
    "Africa": "Africa",
    "North America": "North America",
    "South America": "South America",
    "Oceania": "Oceania",
    "Middle East": "Middle East",
    "asia": "Asia",
    "europe": "Europe",
    "africa": "Africa",
    "north america": "North America",
    "south america": "South America",
    "oceania": "Oceania",
    "middle east": "Middle East",
}



# Metrics
def gini(values):
    """Compute Gini coefficient"""
    values = np.array(values, dtype=float)
    if np.amin(values) < 0:
        raise ValueError("Values cannot be negative")
    if np.sum(values) == 0:
        return 0.0
    values_sorted = np.sort(values)
    n = len(values)
    cumvals = np.cumsum(values_sorted)
    gini_coeff = (n + 1 - 2 * np.sum(cumvals) / cumvals[-1]) / n
    return gini_coeff

def hhi(values):
    """Compute Herfindahl–Hirschman Index (HHI)"""
    values = np.array(values, dtype=float)
    total = np.sum(values)
    shares = values / total
    return np.sum(shares ** 2)

def liveness_coefficient(values):
    """Compute Liveness Coefficient"""
    values_sorted = np.sort(values)[::-1]
    total_value = np.sum(values_sorted)
    for i, v in enumerate(values_sorted):
        if np.sum(values_sorted[:i+1]) >= total_value / 3:
            return (i + 1)

# ------------------------ Data containers ------------------------
@dataclass
class MetricSeries:
    clusters: np.ndarray
    total_dist: np.ndarray
    avg_nnd: np.ndarray
    nni: np.ndarray


# ------------------------ I/O helpers ------------------------
def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def _load_json_if_exists(path: Path):
    if path.is_file():
        with open(path, "r") as f:
            return json.load(f)
    return None


def load_slots(path: Path) -> List[Any]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a list-of-slots JSON.")
    return data


def find_slots_file(run_dir: Path) -> Path:
    candidates = [
        run_dir / "data.json",
        run_dir / "slots.json",
        *sorted(run_dir.glob("slots*.json")),
        *sorted(run_dir.glob("**/slots*.json")),
        *sorted(run_dir.glob("**/data.json")),
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(f"No slots JSON found under {run_dir}")


# ------------------------ Metrics ------------------------
def _safe_metrics_for_points(
    points: Sequence[Any],
) -> Tuple[float, float, float, float]:
    if points is None or len(points) <= 1:
        return 0.0, 0.0, 0.0, 0.0
    dm = init_distance_matrix(points, SPACE)
    c = float(cluster_matrix(dm))
    td = float(total_distance(dm))
    nnd = float(average_nearest_neighbor_distance(dm))
    nni = float(nearest_neighbor_index_spherical(dm, SPACE)[0])
    return c, td, nnd, nni


def compute_metrics_per_slot(
    slots: Sequence[Sequence[Any]],
    granularity: int = 1,
) -> MetricSeries:
    n = len(slots)
    clusters = np.zeros(n, dtype=float)
    total_dist = np.zeros(n, dtype=float)
    avg_nnd = np.zeros(n, dtype=float)
    nni = np.zeros(n, dtype=float)

    last = (0.0, 0.0, 0.0, 0.0)
    g = max(1, granularity)
    for i, pts in enumerate(slots):
        if i % g == 0:
            last = _safe_metrics_for_points(pts)
        elif pts is None or len(pts) <= 1:
            last = (0.0, 0.0, 0.0, 0.0)
        clusters[i], total_dist[i], avg_nnd[i], nni[i] = last
    return MetricSeries(clusters, total_dist, avg_nnd, nni)


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


def parse_name(name: str) -> Dict[str, Any]:
    """
    Parse a simulation run name into a dictionary of parameters.
    """
    
    name = name.replace("num_slots", "slots").replace("time_window", "window")
    parts = name.split("_")
    result = {}
    i = 0
    while i < len(parts) - 1:
        key = parts[i]
        value = parts[i+1]
        try:
            if "." in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            pass
        result[key] = value
        i += 2

    return result


def parse_profit(df):
    value = []
    df["macro-region"] = df["gcp_region"].apply(lambda x: region_maps.get(x.split('-')[0].lower(), 'Other'))
    for slot, slot_df in df.groupby("slot"):
        array = []
        for macro_region, mr_df in slot_df.groupby("macro-region"):
            total_profit = mr_df["mev_offer"].max()
            array.append(total_profit)
        
        # variance
        value.append({
            "slot": slot,
            "variance": pd.Series(array).var(),
            "std": pd.Series(array).std(),
            "mean": pd.Series(array).mean(),
            "cv": pd.Series(array).std() / pd.Series(array).mean() if pd.Series(array).mean() != 0 else 0,
            "gini": gini(pd.Series(array).values)
        })
    return pd.DataFrame(value)
    




# ------------------------ Dash-extras computation ------------------------
def _sum_list_or_zero(xs):
    return float(sum(xs)) if xs else 0.0


def compute_extras_for_slot_series(
    run_dir: Path,
    slots: Sequence[Sequence[Any]],
) -> Dict[str, Any]:
    n = len(slots)
    mev_by_slot = _load_json_if_exists(run_dir / "mev_by_slot.json")
    attest_by_slot = _load_json_if_exists(run_dir / "attest_by_slot.json")
    failed_blocks = _load_json_if_exists(run_dir / "failed_block_proposals.json")
    proposal_time_by_slot = _load_json_if_exists(run_dir / "proposal_time_by_slot.json")
    relay_names = _load_json_if_exists(run_dir / "relay_names.json")
    relay_data = _load_json_if_exists(run_dir / "relay_data.json")

    if len(relay_names) == 0:
        relay_names = _load_json_if_exists(run_dir / "info_names.json")
        relay_data = _load_json_if_exists(run_dir / "info_data.json")

    mev_hist = None
    if isinstance(mev_by_slot, list) and len(mev_by_slot) >= n:
        mev_hist = [_sum_list_or_zero(mev_by_slot[i]) for i in range(n)]

    attest_hist = None
    if isinstance(attest_by_slot, list) and len(attest_by_slot) >= n:
        attest_hist = [_sum_list_or_zero(attest_by_slot[i]) for i in range(n)]

    failed_hist = None
    if isinstance(failed_blocks, list) and len(failed_blocks) >= n:
        failed_hist = [float(failed_blocks[i]) for i in range(n)]

    proposal_hist = None
    if isinstance(proposal_time_by_slot, list) and len(proposal_time_by_slot) >= n:
        proposal_hist = [
            _sum_list_or_zero([t for t in proposal_time_by_slot[i] if t > 0])
            for i in range(n)
        ]

    # relay_series: Dict[str, List[float]] = {}
    # if isinstance(relay_data, list) and n > 0:
    #     relay_positions = []
    #     if relay_data:
    #         if (
    #             isinstance(relay_data[0], list)
    #             and relay_data[0]
    #             and isinstance(relay_data[0][0], (int, float))
    #         ):
    #             relay_positions = [relay_data]
    #         else:
    #             relay_positions = relay_data[0] if relay_data else []
    #     if not relay_positions:
    #         relay_names = []
    #     elif not relay_names or len(relay_names) != len(relay_positions):
    #         relay_names = [f"relay{i+1}" for i in range(len(relay_positions))]
    #     for rname in relay_names:
    #         relay_series[rname] = [0.0] * n
    #     for i in range(n):
    #         pts = slots[i]
    #         if not pts or not relay_positions:
    #             continue
    #         for j, rpos in enumerate(relay_positions):
    #             try:
    #                 avg_d = (
    #                     float(np.mean([SPACE.distance(pt, rpos) for pt in pts]))
    #                     if pts
    #                     else 0.0
    #                 )
    #             except Exception:
    #                 avg_d = 0.0
    #             relay_series[relay_names[j]][i] = avg_d

    # import IPython; IPython.embed(colors="neutral")  # noqa: E402   

    return {
        "mev": mev_hist,
        "attest": attest_hist,
        "failed": failed_hist,
        "proposal": proposal_hist,
        # "relay": relay_series,
    }


# ------------------------ Countries / Regions ------------------------
def load_region_to_country(data_dir: Path) -> Dict[str, str]:
    """Parse gcp_regions.csv mapping Region -> country (last token of 'location')."""
    import csv

    csv_path = data_dir / "gcp_regions.csv"
    if not csv_path.is_file():
        return {}
    mapping = {}
    with open(csv_path, newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            region = row.get("Region") or row.get("region") or ""
            loc = row.get("location") or row.get("Location") or ""
            if not region:
                continue
            country = loc.split(",")[-1].strip() if "," in loc else loc.strip()
            mapping[region] = country or "Unknown"
    return mapping


def compute_metrics(run_dir: Path, data_dir: Path) -> Dict[str, Any]:
    region_counts_per_slot = _load_json_if_exists(
        run_dir / "region_counter_per_slot.json"
    )
    validator_agent_countries = {}
    validator_agent_macro_regions = {}
    region_df = pd.read_csv(f"{data_path}/gcp_regions.csv")
    region_to_country = {}
    for region, city in zip(region_df["Region"], region_df["location"]):
        region_to_country[region] = city.split(",")[-1].strip() if "," in city else city.strip()
    
    for slot, region_list in region_counts_per_slot.items():
        country_counter = defaultdict(int)
        macro_region_counter = defaultdict(int)
        for region, count in region_list:
            country = region_to_country.get(region, "Unknown")
            country_counter[country] += count
            prefix = region.split('-')[0].lower()
            macro_region = region_maps.get(prefix, 'Other')
            macro_region_counter[macro_region] += count
        
        validator_agent_macro_regions[slot] = Counter(macro_region_counter).most_common()
        validator_agent_countries[slot] = Counter(country_counter).most_common()

    region_metrics = []
    country_metrics = []
    macro_region_metrics = []

    initial_num_of_regions = region_df.shape[0]
    initial_num_of_countries = len(set(region_to_country.values()))
    initial_num_of_macro_regions = len(set(region_maps.values()))

    for slot in region_counts_per_slot:
        count_values = np.array([count for _, count in region_counts_per_slot[slot]], dtype=int)
        count_values = np.append(count_values, [0]*(initial_num_of_regions - len(count_values)))
        gini_value = gini(count_values)
        hhi_value = hhi(count_values)
        live_coeff = liveness_coefficient(count_values)
        region_metrics.append((int(slot), gini_value, hhi_value, live_coeff))
    
    # import IPython; IPython.embed(colors="neutral")  # noqa: E402
    
    for slot in validator_agent_countries:
        count_values = np.array([count for _, count in validator_agent_countries[slot]], dtype=int)
        count_values = np.append(count_values, [0]*(initial_num_of_countries - len(count_values)))
        gini_value = gini(count_values)
        hhi_value = hhi(count_values)
        live_coeff = liveness_coefficient(count_values)
        
        country_metrics.append((int(slot), gini_value, hhi_value, live_coeff))

    for slot in validator_agent_macro_regions:
        count_values = np.array([count for _, count in validator_agent_macro_regions[slot]], dtype=int)
        count_values = np.append(count_values, [0]*(initial_num_of_macro_regions - len(count_values)))
        gini_value = gini(count_values)
        hhi_value = hhi(count_values)
        live_coeff = liveness_coefficient(count_values)

        # import IPython; IPython.embed(colors="neutral")  # noqa: E402
        macro_region_metrics.append((int(slot), gini_value, hhi_value, live_coeff))


    region_df = pd.DataFrame(sorted(region_metrics, key=lambda x: x[0]), columns=["slot", "gini", "hhi", "liveness"])
    country_df = pd.DataFrame(sorted(country_metrics, key=lambda x: x[0]), columns=["slot", "gini", "hhi", "liveness"])
    macro_region_df = pd.DataFrame(sorted(macro_region_metrics, key=lambda x: x[0]), columns=["slot", "gini", "hhi", "liveness"])

    return (region_df, country_df, macro_region_df)

def compute_country_histograms(run_dir: Path, data_dir: Path) -> Dict[str, Any]:
    """
    Returns:
      {
        "overall": Counter-like dict country->count over all slots,
        "final": dict country->count at final slot,
        "top_overall": list[(country,count)] top-15,
        "top_final": list[(country,count)] top-15,
      }
    """
    region_counts_per_slot = _load_json_if_exists(
        run_dir / "region_counter_per_slot.json"
    )
    if not isinstance(region_counts_per_slot, dict) or not region_counts_per_slot:
        return {}
    region_to_country = load_region_to_country(data_dir)
    # overall
    overall: Dict[str, int] = {}
    final: Dict[str, int] = {}

    # Keys are slot indices as strings; iterate sorted by int
    slots_sorted = sorted(region_counts_per_slot.keys(), key=lambda s: int(s))
    for slot in slots_sorted:
        pairs = region_counts_per_slot[slot]  # list of [region, count]
        agg: Dict[str, int] = {}
        for region, count in pairs:
            country = region_to_country.get(region, "Unknown")
            agg[country] = agg.get(country, 0) + int(count)
            overall[country] = overall.get(country, 0) + int(count)
        final = agg  # last assignment ends up as final slot
    print(final)

    # sort top-15
    top_overall = sorted(overall.items(), key=lambda kv: kv[1], reverse=True)[:15]
    top_final = sorted(final.items(), key=lambda kv: kv[1], reverse=True)[:15]
    return {
        "overall": overall,
        "final": final,
        "top_overall": top_overall,
        "top_final": top_final,
    }


# ------------------------ Plotting helpers ------------------------


def single_line(ax: plt.Axes, data_df: pd.DataFrame, x_col: str, y_col: str, hue: str, ylabel: str, title: str = "", legend: bool = True, xlabel: str = "Slot"):
    sns.lineplot(
        data=data_df,
        x=x_col,
        y=y_col,
        hue=hue,
        style=hue,
        dashes=[(None, None), (None, None), (1,1), (1,1)],
        ax=ax,
        lw=8.0,
        legend=legend
    )
    ax.set_xlabel(xlabel, fontsize=40)
    ax.set_ylabel(ylabel, fontsize=40)
    ax.set_xlim(0, data_df[x_col].max()+1)
    # plt.xticks(fontsize=28)
    # plt.yticks(fontsize=28)
    if xlabel is not None:
        ax.tick_params(axis="x", labelsize=36)
    else:
        ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    # else:
        # ax.set_xlabel("", fontsize=36)
    # ax.tick_params(axis="x", labelsize=36)
    ax.tick_params(axis="y", labelsize=36)
    # ax.set_xticks(fontsize=28)
    # ax.set_yticks(fontsize=28)
    # plt.legend(title=None, fontsize=20)
    # ax.set_title(title)
    # return ax


def save_fig(fig: plt.Figure, outfile: Path, fmt: str, dpi: int):
    outfile = outfile.with_suffix(f".{fmt}")
    fig.tight_layout()
    fig.savefig(outfile, dpi=dpi)
    if fmt.lower() != "pdf":
        fig.savefig(outfile.with_suffix(".pdf"))
    print("✓", outfile)


def bar_on(ax: plt.Axes, items: List[Tuple[str, float]], title: str, xlabel: str):
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
    ax.set_ylabel("Validators")



def sns_bar_on(ax: plt.Axes, data_df: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, hue: str = None):
    sns.barplot(
        ax=ax,
        data=data_df,
        x=x_col,
        y=y_col,
        hue=hue,
    )
    labels = data_df[x_col].tolist()
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("# Validators")



# ------------------------ Per-run worker ------------------------
def analyze_one(
    slots_path: Path,
    outdir: Path,
    *,
    granularity: int,
    rolling: int,
    every: int,
    data_dir: Path,
):
    slots = load_slots(slots_path)
    n = len(slots)

    # metrics = compute_metrics_per_slot(slots, granularity=granularity)
    # if rolling and rolling > 1:
    #     metrics = MetricSeries(
    #         clusters=maybe_rolling(metrics.clusters, rolling),
    #         total_dist=maybe_rolling(metrics.total_dist, rolling),
    #         avg_nnd=maybe_rolling(metrics.avg_nnd, rolling),
    #         nni=maybe_rolling(metrics.nni, rolling),
    #     )
    
    x_full = np.arange(1, n + 1)
    keep = np.arange(0, n, max(1, every))

    # metrics_df = pd.DataFrame({
    #     "slot": x_full[keep],
    #     "clusters": metrics.clusters[keep],
    #     "total_dist": metrics.total_dist[keep],
    #     "avg_nnd": metrics.avg_nnd[keep],
    #     "nni": metrics.nni[keep],
    # })

    # -------- Extras (MEV/attest/failed/proposal + relays) --------
    run_dir = outdir.parent
    extras = compute_extras_for_slot_series(run_dir, slots)
    extras_df = pd.DataFrame({
        "slot": x_full[keep],
        "mev": [extras["mev"][i] if extras["mev"] else 0.0 for i in keep],
        "attest": [extras["attest"][i] if extras["attest"] else 0.0 for i in keep],
        "failed": [extras["failed"][i] if extras["failed"] else 0.0 for i in keep],
        "proposal": [extras["proposal"][i] if extras["proposal"] else 0.0 for i in keep],
    })

    # relay_extras = extras.get("relay", {}) or extras.get("info", {}) or {}
    # relay_extras_dfs = []
    # for rname, series in relay_extras.items():
    #     df = pd.DataFrame({
    #         "slot": x_full[keep],
    #         "relay": rname,
    #         "avg_distance": [series[i] if series else 0.0 for i in keep],
    #     })
    #     relay_extras_dfs.append(df)
    
    # -------- Countries (overall + final slot) --------
    countries = compute_country_histograms(run_dir, data_dir)

    eval_metrics = compute_metrics(run_dir, data_dir)

    profits_df = parse_profit(pd.read_csv(run_dir / "region_profits.csv"))
    
    return None, extras_df, None, countries, eval_metrics, profits_df
    


# ------------------------ CLI ------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Plot spatial metrics for one or many simulation runs (+ extras, countries, panels)."
    )
    # batch mode
    ap.add_argument(
        "-b",
        "--batch",
        help="Directory whose immediate subfolders are simulation runs.",
    )
    ap.add_argument(
        "--outsubdir",
        default="figures",
        help="Subfolder created under each run to store outputs.",
    )
    # shared knobs
    ap.add_argument(
        "-g",
        "--granularity",
        type=int,
        default=10,
        help="Recompute metrics every g-th slot",
    )
    ap.add_argument(
        "--rolling",
        type=int,
        default=1,
        help="Centered rolling smoothing window (odd recommended)",
    )
    ap.add_argument(
        "--every", type=int, default=1, help="Plot every k-th point (downsample)"
    )
    ap.add_argument(
        "--list",
        type=str,
        nargs='+',
        help="List of run directories to compare (comma-separated)",
    )
    ap.add_argument(
        "--fmt",
        default="pdf",
        choices=["pdf"],
        help="Output figure format",
    )
    ap.add_argument("--dpi", type=int, default=300, help="Raster DPI (if png)")
    ap.add_argument("--size", default="10x4", help="Figure size in inches, e.g., 12x4")
    ap.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing gcp_regions.csv for country mapping",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir)

    # if args.batch:
    #     root = Path(args.batch)
    #     if not root.is_dir():
    #         raise SystemExit(f"--batch path not a directory: {root}")
    #     run_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    #     runtime_configs = [parse_name(rd.name) for rd in run_dirs if rd.is_dir() and rd.name.startswith("num_")]

    if args.list:
        print(f"Comparing runs: {args.list}")
        run_dirs = [Path(p.strip()) for p in args.list if Path(p.strip()).is_dir()]
        # common_configs = {}
        # for key in runtime_configs[0].keys():
        #     if all(rc.get(key) == runtime_configs[0].get(key) for rc in runtime_configs):
        #         common_configs[key] = runtime_configs[0][key]
        common_configs_str = " | "
        # print(f"Common configs: {common_configs}")
        # common_configs_str = ",".join(f"{k}={v}" for k, v in common_configs.items())
        print(f"Common configs string: {common_configs_str}")

        # if not run_dirs:
        #     raise SystemExit(f"No subfolders under {root}")
        # print(f"Found {len(run_dirs)} runs under {root}")
        total_metrics = []
        total_extras = []
        total_relay = []
        total_countries = []
        total_profits = []

        total_region_metrics = []
        total_country_metrics = []
        total_macro_region_metrics = []
        # for rd, unique_name in zip(run_dirs, ["latency-aligned (SSP)", "latency-misaligned (SSP)", "latency-aligned (MSP)", "latency-misaligned (MSP)"]):
        # for rd, unique_name in zip(run_dirs, ["latency-aligned (SSP)", "latency-misaligned (SSP)", "latency-aligned (MSP)", "latency-misaligned (MSP)"]):
        for rd, unique_name in zip(run_dirs, [r"$\Delta=12s$ (SSP)", r"$\Delta=6s$ (SSP)", r"$\Delta=12s$ (MSP)", r"$\Delta=6s$ (MSP)"]):
            try:
                print(f"\nProcessing {rd} ...")
                slots_path = find_slots_file(rd)
                config = parse_name(rd.name)

                outdir = rd / args.outsubdir
                print(
                    f"\n→ {rd.name} | slots: {slots_path.relative_to(rd)} | out: {outdir.relative_to(rd)}"
                )
                metrics_df, extras_df, relay_extras_dfs, countries, eval_metrics_dfs, profits_df = analyze_one(
                    slots_path,
                    outdir,
                    granularity=max(1, args.granularity),
                    rolling=args.rolling,
                    every=max(1, args.every),
                    data_dir=data_dir,
                )

                region_df, country_df, macro_df = eval_metrics_dfs

                # unique_name = ",".join(f"{k}={v}" for k, v in config.items() if k not in common_configs)
                # metrics_df["name"] = unique_name
                profits_df["name"] = unique_name
                extras_df["name"] = unique_name
                # relay_extras_dfs["name"] = unique_name
                countries["name"] = unique_name

                region_df["name"] = unique_name
                country_df["name"] = unique_name
                macro_df["name"] = unique_name
                total_region_metrics.append(region_df)
                total_country_metrics.append(country_df)
                total_macro_region_metrics.append(macro_df)
                total_profits.append(profits_df)

                # total_metrics.append(metrics_df)
                total_extras.append(extras_df)
                # total_relay.append(relay_extras_dfs)
                total_countries.append(countries)
                print(f"✓ Processed {rd.name} with {len(total_extras)} slots")
            except Exception as e:
                print(f"✗ Skipping {rd}: {e}")
        

        # total_metrics_df = pd.concat(total_metrics, ignore_index=True)
        total_region_metrics_df = pd.concat(total_region_metrics, ignore_index=True)
        total_country_metrics_df = pd.concat(total_country_metrics, ignore_index=True)
        total_macro_region_metrics_df = pd.concat(total_macro_region_metrics, ignore_index=True)
        total_extras_df = pd.concat(total_extras, ignore_index=True)
        total_profits_df = pd.concat(total_profits, ignore_index=True)
        # total_relay_df = pd.concat(total_relay)
        total_countries_df = pd.DataFrame(total_countries)

        try:
            w_str, h_str = args.size.lower().split("x")
            fig_w, fig_h = float(w_str), float(h_str)
        except Exception:
            fig_w, fig_h = 16.0, 9.0

        outdir = args.outsubdir
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        import IPython; IPython.embed(colors="neutral")  # noqa: E402
        # for y, y_label in zip(["clusters", "total_dist", "avg_nnd", "nni"], ["# Clusters", "Total Distance", "Avg. NND", "NNI"]):
        #     fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        #     single_line(
        #         ax,
        #         total_metrics_df,
        #         "slot",
        #         y,
        #         "name",
        #         y_label,
        #         common_configs_str
        #     )
        #     save_fig(fig, outdir / f"{y}_per_slot", args.fmt, args.dpi)
        #     plt.close(fig)

        fig, axes = plt.subplots(1,3, figsize=(fig_w*3, fig_h), sharey=False)
        for idx, y, y_label in zip(range(3), ["gini", "hhi", "liveness"], ["Gini$_{g}$", "HHI$_{g}$", "LC$_{g}$"]):
            single_line(
                axes[idx],
                total_region_metrics_df,
                "slot",
                y,
                "name",
                y_label,
                f"Regions | {common_configs_str}",
                False if idx < 2 else True
            )
        handles, labels = axes[-1].get_legend_handles_labels()
        print(handles, labels)
        axes[-1].legend_.remove()
        # plt.tight_layout(rect=[0, 0, 1, 0.8])

        fig.legend(handles, labels, loc="upper center", fontsize=36, ncol=5, title=None, bbox_to_anchor=(0.5, 1.05), 
            framealpha=0, facecolor="none", edgecolor="none", columnspacing=0.5)
        plt.savefig(outdir / f"regions_metrics_per_slot.{args.fmt}", dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(2,2, figsize=(fig_w*2, fig_h*2), sharey=False, sharex=True)
        axes = axes.flatten()
        for idx, y, y_label in zip(range(3), ["gini", "liveness", "hhi"], ["Gini$_{g}$",  "LC$_{g}$", "HHI$_{g}$"]):
            ax = axes[idx]
            single_line(
                axes[idx],
                total_macro_region_metrics_df,
                "slot",
                y,
                "name",
                y_label,
                f"Macro-regions | {common_configs_str}",
                False,
                "Slot" if idx == 2 else None
            )
        single_line(
            axes[-1],
            total_profits_df,
            "slot",
            "cv",
            "name",
            "CV$_{g}$",
            f"Macro-regions Profit CV | {common_configs_str}",
            True,
            "Slot"
        )

        plt.subplots_adjust(hspace=0.1, wspace=0.3)
        handles, labels = axes[-1].get_legend_handles_labels()
        print(handles, labels)
        axes[-1].legend_.remove()
        # plt.tight_layout(rect=[0, 0, 1, 0.8])

        fig.legend(handles, labels, loc="upper center", fontsize=40, ncol=2, title=None, bbox_to_anchor=(0.5, 1.04), 
            framealpha=0, facecolor="none", edgecolor="none", columnspacing=5)
        plt.savefig(outdir / f"macro_regions_metrics_per_slot.{args.fmt}", dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        
        
        # plt.tight_layout()
        # save_fig(fig, outdir / f"regions_metrics_per_slot", args.fmt, args.dpi)
        # plt.close(fig)

            # save_fig(fig, outdir / f"regions_{y}_per_slot", args.fmt, args.dpi)
            # plt.close(fig)


        for y, y_label in zip(["gini", "hhi", "liveness"], ["Gini Coefficient", "HHI", "Liveness Coefficient"]):
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            single_line(
                ax,
                total_region_metrics_df,
                "slot",
                y,
                "name",
                y_label,
                f"Regions | {common_configs_str}",
                False
            )
            save_fig(fig, outdir / f"regions_{y}_per_slot", args.fmt, args.dpi)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            single_line(
                ax,
                total_country_metrics_df,
                "slot",
                y,
                "name",
                y_label,
                f"Countries | {common_configs_str}"
            )
            save_fig(fig, outdir / f"countries_{y}_per_slot", args.fmt, args.dpi)
            plt.close(fig)


        for y, y_label in zip(["mev", "attest", "failed", "proposal"], ["MEV", "Attestation", "Failed Proposals", "Proposal Time"]):
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            single_line(
                ax,
                total_extras_df,
                "slot",
                y,
                "name",
                y_label,
                common_configs_str
            )
            save_fig(fig, outdir / f"{y}_per_slot", args.fmt, args.dpi)
            plt.close(fig)

    
        # countries
        if not total_countries_df.empty:
            for col in ["top_overall", "top_final"]:
                all_items = []
                for _, row in total_countries_df.iterrows():
                    name = row.get("name", "unknown")
                    items = row.get(col, [])
                    for country, count in items:
                        all_items.append((name, country, count))
                if all_items:
                    countries_df = pd.DataFrame(all_items, columns=["name", "country", "count"])
                    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                    sns_bar_on(
                        ax,
                        countries_df,
                        "country",
                        "count",
                        title=f"Top Countries ({col.replace('_', ' ').title()}) | {common_configs_str}",
                        xlabel="Country",
                        hue="name",
                    )
                    save_fig(fig, outdir / f"countries_{col}", args.fmt, args.dpi)

                plt.close(fig)

        print(f"✓ Processed {len(run_dirs)} runs, outputs saved to {outdir}")

if __name__ == "__main__":
    main()
