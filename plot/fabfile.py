# fabfile.py
from fabric import task
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import os
import re


from collections import defaultdict, Counter
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CUR_DIR, "..", "data")
FIGURE_DIR = os.path.join(CUR_DIR, "..", "figure")
OUTPUT_DIR = os.path.join(CUR_DIR, "..", "output")

_CONTINENT_RULES = [
    (r"^us-|^northamerica-", "North America"),
    (r"^southamerica-",      "South America"),
    (r"^europe-",            "Europe"),
    (r"^asia-",              "Asia"),
    (r"^australia-",         "Oceania"),
    (r"^me-",                "Middle East"),
    (r"^africa-",            "Africa"),
]


def to_continent(region: str) -> str:
    for pat, name in _CONTINENT_RULES:
        if re.match(pat, region):
            return name
    return "Other"


def _load_json_if_exists(path: Path):
    if path.is_file():
        with open(path, "r") as f:
            return json.load(f)
    return None


def _with_seed_suffix(path: str, seed=None) -> str:
    if seed is None:
        return path
    return f"{path}_seed_{seed}"


def _format_latency_std_dev_ratio(latency_std_dev_ratio) -> str:
    return format(float(latency_std_dev_ratio), "g")


def _with_run_suffix(path: str, seed=None, latency_std_dev_ratio=None) -> str:
    path = _with_seed_suffix(path, seed)
    if latency_std_dev_ratio is not None:
        path = f"{path}_latstd_{_format_latency_std_dev_ratio(latency_std_dev_ratio)}"
    return path


def _resolve_default_latency_ratio_path(path: str, seed=None) -> str:
    """Prefer explicit `_latstd_0.5` outputs when present, otherwise fall back to the legacy default path."""
    explicit_path = _with_run_suffix(path, seed, 0.5)
    if os.path.exists(explicit_path):
        return explicit_path
    return _with_run_suffix(path, seed, None)


def _baseline_variation_path(model: str, ratio: float, seed=None) -> str:
    base_path = os.path.join(OUTPUT_DIR, "baseline", model, "validators_1000_slots_10000_cost_0.0")
    if ratio == 0.5:
        return _resolve_default_latency_ratio_path(base_path, seed)
    return _with_run_suffix(base_path, seed, ratio)


def _figure_output_path(filename: str, seed=None, latency_std_dev_ratio=None) -> str:
    stem, ext = os.path.splitext(filename)
    return os.path.join(FIGURE_DIR, f"{_with_run_suffix(stem, seed, latency_std_dev_ratio)}{ext}")


def _extract_seed_from_run_dir(name: str):
    match = re.search(r"_seed_(\d+)(?:_|$)", name)
    if match is None:
        return None
    return int(match.group(1))


def _collect_seeded_run_dirs(base_path: str, latency_std_dev_ratio=None, max_runs=20) -> list[Path]:
    """Collect baseline run folders across seeds, optionally filtering by latency std-dev ratio."""
    base = Path(base_path)
    parent = base.parent
    prefix = f"{base.name}_seed_"
    explicit_matches = {}
    fallback_matches = {}
    ratio_tag = (
        _format_latency_std_dev_ratio(latency_std_dev_ratio)
        if latency_std_dev_ratio is not None
        else None
    )

    if not parent.exists():
        raise FileNotFoundError(f"Parent directory does not exist: {parent}")

    for candidate in parent.iterdir():
        if not candidate.is_dir() or not candidate.name.startswith(prefix):
            continue

        seed = _extract_seed_from_run_dir(candidate.name)
        if seed is None:
            continue

        has_explicit_ratio = "_latstd_" in candidate.name
        if ratio_tag is None:
            if not has_explicit_ratio:
                fallback_matches[seed] = candidate
            continue

        if candidate.name.endswith(f"_latstd_{ratio_tag}"):
            explicit_matches[seed] = candidate
        elif ratio_tag == "0.5" and not has_explicit_ratio:
            fallback_matches.setdefault(seed, candidate)

    if ratio_tag is None:
        selected = fallback_matches
    elif ratio_tag == "0.5":
        selected = dict(fallback_matches)
        selected.update(explicit_matches)
    else:
        selected = explicit_matches

    selected_dirs = [selected[seed] for seed in sorted(selected)]
    if max_runs is not None:
        selected_dirs = selected_dirs[:max_runs]

    if not selected_dirs:
        raise FileNotFoundError(
            f"No run directories found for base path '{base_path}' with latency_std_dev_ratio={latency_std_dev_ratio}"
        )

    return selected_dirs


def _resolve_run_path(base_path: str, seed=None, latency_std_dev_ratio=None) -> Path:
    """Resolve one run directory, preserving the 0.5 fallback to legacy default paths."""
    if latency_std_dev_ratio == 0.5:
        return Path(_resolve_default_latency_ratio_path(base_path, seed))
    return Path(_with_run_suffix(base_path, seed, latency_std_dev_ratio))


def _resolve_run_group(base_path: str, seed=None, latency_std_dev_ratio=None, max_runs=20) -> list[Path]:
    """Return either one run path (when seed is provided) or a seeded run group for averaging."""
    if seed is not None:
        return [_resolve_run_path(base_path, seed, latency_std_dev_ratio)]
    return _collect_seeded_run_dirs(
        base_path,
        latency_std_dev_ratio=latency_std_dev_ratio,
        max_runs=max_runs,
    )


def load_folder(folder_path: Path) -> pd.DataFrame:
    data = _load_json_if_exists(folder_path / "region_counter_per_slot.json")

    total_validators = sum([i[1] for i in data['0']])
    data = {int(k): v for k, v in data.items()}

    continent_data = []
    for slot, region_counts in data.items():
        continent_count = defaultdict(int)
        for region, count in region_counts:
            continent = to_continent(region)
            continent_count[continent] += count

        for continent, count in list(continent_count.items()):
            p = 100 * count / total_validators
            continent_data.append({
                "slot": slot,
                "continent": continent,
                "count": count,
                "percentage": p
            })

    return pd.DataFrame(continent_data)


def plot_continent_distribution(folder_paths, xlabels, ylabels, output_path, subplots=(2,2)):
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(subplots[0], subplots[1], figsize=(34, 8*subplots[0]-1), dpi=300, sharey=True)

    hue_order = ['North America', 'Europe', 'Asia', 'Middle East', 'Oceania', 'South America', 'Africa']

    continent_dfs = [load_folder(Path(fp)) for fp in folder_paths]
    new_axes = []

    for continent_df, xlabel, ylabel, ax in zip(continent_dfs, xlabels, ylabels, axes.flatten()):
        new_ax = sns.lineplot(
            data=continent_df,
            x='slot',
            y='percentage',
            hue='continent',
            style='continent',
            lw=9.0,
            hue_order=hue_order,
            style_order=hue_order,
            ax=ax,
            markers=True,
            markevery=2000, 
            markersize=20
        )
        new_ax.set_xlabel(xlabel, fontsize=42)
        if len(new_axes) % 2 == 0: # only set ylabel for left column
            new_ax.set_ylabel(ylabel, fontsize=42)
        new_ax.legend_.remove()
        new_ax.set_xlim(0, max(continent_df['slot'])+1)
        new_ax.tick_params(labelsize=42)
        new_axes.append(new_ax)

    handles, labels = new_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title=None, fontsize=42, ncol=7, loc='upper center', bbox_to_anchor=(0.5, 1.16-0.04*subplots[0]), framealpha=0, facecolor="none", edgecolor="none", columnspacing=0.5)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def plot_continent_distribution_mean(folder_path_groups, xlabels, ylabels, output_path, subplots=(2,2)):
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(subplots[0], subplots[1], figsize=(34, 8*subplots[0]-1), dpi=300, sharey=True)

    hue_order = ['North America', 'Europe', 'Asia', 'Middle East', 'Oceania', 'South America', 'Africa']
    new_axes = []

    for folder_paths, xlabel, ylabel, ax in zip(folder_path_groups, xlabels, ylabels, axes.flatten()):
        continent_runs = []
        for folder_path in folder_paths:
            continent_df = load_folder(Path(folder_path)).copy()
            continent_df["run_id"] = Path(folder_path).name
            continent_runs.append(continent_df)

        mean_continent_df = (
            pd.concat(continent_runs, ignore_index=True)
            .groupby(["slot", "continent"], as_index=False)[["count", "percentage"]]
            .mean()
        )

        new_ax = sns.lineplot(
            data=mean_continent_df,
            x='slot',
            y='percentage',
            hue='continent',
            style='continent',
            lw=9.0,
            hue_order=hue_order,
            style_order=hue_order,
            ax=ax,
            markers=True,
            markevery=2000,
            markersize=20
        )
        new_ax.set_xlabel(xlabel, fontsize=42)
        if len(new_axes) % 2 == 0:
            new_ax.set_ylabel(ylabel, fontsize=42)
        new_ax.legend_.remove()
        new_ax.set_xlim(0, max(mean_continent_df['slot'])+1)
        new_ax.tick_params(labelsize=42)
        new_axes.append(new_ax)

    handles, labels = new_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title=None, fontsize=42, ncol=7, loc='upper center', bbox_to_anchor=(0.5, 1.16-0.04*subplots[0]), framealpha=0, facecolor="none", edgecolor="none", columnspacing=0.5)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


# metrics
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
    return np.sum(shares**2)


def liveness_coefficient(values):
    """Compute Liveness Coefficient"""
    values_sorted = np.sort(values)[::-1]
    total_value = np.sum(values_sorted)
    for i, v in enumerate(values_sorted):
        if np.sum(values_sorted[: i + 1]) >= total_value / 3:
            return i + 1


def compute_metrics(folder_path: Path) -> Optional[List[pd.DataFrame]]:
    region_counts_path = folder_path / "region_counter_per_slot.json"
    region_counts_per_slot = _load_json_if_exists(region_counts_path)
    if region_counts_per_slot is None:
        print(f"Missing data file: {region_counts_path}")
        return None
    validator_agent_countries = {}
    validator_agent_continents = {}
    region_df = pd.read_csv(f"{DATA_DIR}/gcp_regions.csv")
    region_to_country = {}
    for region, city in zip(region_df["Region"], region_df["location"]):
        region_to_country[region] = (
            city.split(",")[-1].strip() if "," in city else city.strip()
        )

    for slot, region_list in region_counts_per_slot.items():
        country_counter = defaultdict(int)
        continent_counter = defaultdict(int)
        for region, count in region_list:
            country = region_to_country.get(region, "Unknown")
            continent = to_continent(region)
            country_counter[country] += count
            continent_counter[continent] += count

        validator_agent_countries[slot] = Counter(
            country_counter
        ).most_common()

        validator_agent_continents[slot] = Counter(
            continent_counter
        ).most_common()


    metrics_dfs = []

    initial_num_of_regions = region_df.shape[0]
    initial_num_of_countries = len(set(region_to_country.values()))
    initial_num_of_continents = len(_CONTINENT_RULES)

    for counts, initial_num in zip(
        [region_counts_per_slot, validator_agent_countries, validator_agent_continents],
        [initial_num_of_regions, initial_num_of_countries, initial_num_of_continents],
    ):
        metrics = []
        for slot in counts:
            count_values = np.array(
                [count for _, count in counts[slot]], dtype=int
            )
            count_values = np.append(
                count_values, [0] * (initial_num - len(count_values))
            )
            gini_value = gini(count_values)
            hhi_value = hhi(count_values)
            live_coeff = liveness_coefficient(count_values)
            metrics.append((int(slot), gini_value, hhi_value, live_coeff))
        
        metrics_dfs.append(pd.DataFrame(
            sorted(metrics, key=lambda x: x[0]),
            columns=["slot", "gini", "hhi", "liveness"],
        ))
        
    return metrics_dfs


def parse_profit(df):
    value = []
    df["continent"] = df["gcp_region"].apply(lambda x: to_continent(x))
    for slot, slot_df in df.groupby("slot"):
        array = []
        for _, mr_df in slot_df.groupby("continent"):
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


def single_line(ax: plt.Axes, data_df: pd.DataFrame, x_col: str, y_col: str, hue: str, ylabel: str, legend: bool = True, xlabel: str = "Slot", dashes=None):
    if dashes is None:
        count = data_df[hue].nunique() // 2
        dashes = [(None, None)] * count + [(2, 2)] * count
    
    sns.lineplot(
        data=data_df,
        x=x_col,
        y=y_col,
        hue=hue,
        style=hue,
        dashes=dashes,
        ax=ax,
        lw=8.0,
        markers=True,
        markevery=2000, 
        markersize=20,
        legend=legend
    )
    ax.set_xlabel(xlabel, fontsize=40)
    ax.set_ylabel(ylabel, fontsize=40)
    if xlabel == "Slot":
        ax.set_xlim(0, data_df[x_col].max()+1)
    if xlabel is not None:
        ax.tick_params(axis="x", labelsize=36)
    else:
        ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis="y", labelsize=36)


def plot_comparision(folder_paths, names, output_path, figsize=(25, 12), desired_order=None, normalized=False, ncol=None, columnspacing=None, h_offset=0.04, dashes=None):
    total_metrcis = []
    total_profits = []

    if ncol is None:
        ncol = len(names)//2 if len(names) > 2 else 2

    if columnspacing is None:
        columnspacing = round(10.5-1.25*len(names), 1)

    for folder_path, name in zip(folder_paths, names):
        folder_path = Path(folder_path)
        eval_metrics = compute_metrics(folder_path)

        profits_df = parse_profit(pd.read_csv(folder_path / "region_profits.csv"))

        region_metrics, country_metrics, continent_metrics = eval_metrics

        region_metrics["name"] = name
        country_metrics["name"] = name
        continent_metrics["name"] = name
        profits_df["name"] = name

        continent_metrics["normalized_slots"] = continent_metrics["slot"] / continent_metrics["slot"].max() if normalized else continent_metrics["slot"]
        profits_df["normalized_slots"] = profits_df["slot"] / profits_df["slot"].max() if normalized else profits_df["slot"]


        total_metrcis.append(continent_metrics)
        total_profits.append(profits_df)

    total_metrics_df = pd.concat(total_metrcis, ignore_index=True)
    total_profits_df = pd.concat(total_profits, ignore_index=True)
    display_order = desired_order if desired_order is not None else names

    _print_final_slot_ci95(
        total_metrics_df,
        total_profits_df,
        display_order,
        normalized=normalized,
        label=Path(output_path).name,
    )

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2,2, figsize=figsize, sharey=False, sharex=True, dpi=300)
    axes = axes.flatten()
    for idx, y, y_label in zip(range(3), ["gini", "liveness", "hhi"], [r"$\mathrm{Gini}_{\mathrm{g}}$",  r"$\mathrm{LC}_{\mathrm{g}}$", r"$\mathrm{HHI}_{\mathrm{g}}$"]):
        single_line(
            axes[idx],
            total_metrics_df,
            "slot" if not normalized else "normalized_slots",
            y,
            "name",
            y_label,
            False,
            ("Slot" if not normalized else "Relative Progress") if idx == 2 else None,
            dashes=dashes,
        )

    # cv    
    single_line(
        axes[-1],
        total_profits_df,
        "slot" if not normalized else "normalized_slots",
        "cv",
        "name",
        r"$\mathrm{CV}_{\mathrm{g}}$",
        True,
        "Slot" if not normalized else "Relative Progress",
        dashes=dashes,
    )

    plt.subplots_adjust(hspace=0.1, wspace=0.3)
    handles, labels = axes[-1].get_legend_handles_labels()

    label_to_handle = dict(zip(labels, handles))
    if desired_order is not None:
        handles = [label_to_handle[label] for label in display_order]
        labels = display_order

    axes[-1].legend_.remove()


    fig.legend(handles, labels, loc="upper center", fontsize=40, ncol=ncol, title=None, bbox_to_anchor=(0.5, 1 + h_offset), framealpha=0, facecolor="none", edgecolor="none", columnspacing=columnspacing)
    plt.savefig(output_path, bbox_inches="tight") 
    plt.close(fig)


def plot_comparision_mean(folder_path_groups, names, output_path, figsize=(25, 12), desired_order=None, normalized=False, ncol=None, columnspacing=None, h_offset=0.04):
    total_metrics_df, total_profits_df = _load_comparison_run_data(
        folder_path_groups,
        names,
        normalized=normalized,
    )

    if ncol is None:
        ncol = len(names)//2 if len(names) > 2 else 2

    if columnspacing is None:
        columnspacing = round(10.5-1.25*len(names), 1)

    x_col = "slot" if not normalized else "normalized_slots"
    display_order = desired_order if desired_order is not None else names

    _print_final_slot_ci95(
        total_metrics_df,
        total_profits_df,
        display_order,
        normalized=normalized,
        label=Path(output_path).name,
    )

    total_metrics_df = (
        total_metrics_df
        .groupby(["name", x_col], as_index=False)[["gini", "hhi", "liveness"]]
        .mean()
    )
    total_profits_df = (
        total_profits_df
        .groupby(["name", x_col], as_index=False)[["cv"]]
        .mean()
    )

    total_metrics_df["name"] = pd.Categorical(total_metrics_df["name"], categories=display_order, ordered=True)
    total_profits_df["name"] = pd.Categorical(total_profits_df["name"], categories=display_order, ordered=True)
    total_metrics_df = total_metrics_df.sort_values(["name", x_col])
    total_profits_df = total_profits_df.sort_values(["name", x_col])

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2,2, figsize=figsize, sharey=False, sharex=True, dpi=300)
    axes = axes.flatten()
    for idx, y, y_label in zip(range(3), ["gini", "liveness", "hhi"], [r"$\mathrm{Gini}_{\mathrm{g}}$",  r"$\mathrm{LC}_{\mathrm{g}}$", r"$\mathrm{HHI}_{\mathrm{g}}$"]):
        single_line(
            axes[idx],
            total_metrics_df,
            x_col,
            y,
            "name",
            y_label,
            False,
            ("Slot" if not normalized else "Relative Progress") if idx == 2 else None
        )

    single_line(
        axes[-1],
        total_profits_df,
        x_col,
        "cv",
        "name",
        r"$\mathrm{CV}_{\mathrm{g}}$",
        True,
        "Slot" if not normalized else "Relative Progress"
    )

    plt.subplots_adjust(hspace=0.1, wspace=0.3)
    handles, labels = axes[-1].get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))
    handles = [label_to_handle[label] for label in display_order]
    labels = display_order
    axes[-1].legend_.remove()

    fig.legend(handles, labels, loc="upper center", fontsize=40, ncol=ncol, title=None, bbox_to_anchor=(0.5, 1 + h_offset), framealpha=0, facecolor="none", edgecolor="none", columnspacing=columnspacing)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _load_comparison_run_data(folder_path_groups, names, normalized=False):
    total_metrics = []
    total_profits = []

    for folder_paths, name in zip(folder_path_groups, names):
        for folder_path in folder_paths:
            folder_path = Path(folder_path)
            eval_metrics = compute_metrics(folder_path)
            if eval_metrics is None:
                continue

            profits_path = folder_path / "region_profits.csv"
            if not profits_path.is_file():
                print(f"Missing data file: {profits_path}")
                continue

            profits_df = parse_profit(pd.read_csv(profits_path))
            _, _, continent_metrics = eval_metrics

            continent_metrics["name"] = name
            continent_metrics["run_id"] = folder_path.name
            profits_df["name"] = name
            profits_df["run_id"] = folder_path.name

            continent_metrics["normalized_slots"] = (
                continent_metrics["slot"] / continent_metrics["slot"].max()
                if normalized
                else continent_metrics["slot"]
            )
            profits_df["normalized_slots"] = (
                profits_df["slot"] / profits_df["slot"].max()
                if normalized
                else profits_df["slot"]
            )

            total_metrics.append(continent_metrics)
            total_profits.append(profits_df)

    if not total_metrics or not total_profits:
        raise FileNotFoundError("No comparison data found. Missing data paths were printed above.")

    return (
        pd.concat(total_metrics, ignore_index=True),
        pd.concat(total_profits, ignore_index=True),
    )


def _compute_uncertainty_bounds(grouped, band_mode: str):
    mean = grouped.mean().sort_index()

    if band_mode == "iqr":
        lower = grouped.quantile(0.25).reindex(mean.index)
        upper = grouped.quantile(0.75).reindex(mean.index)
    elif band_mode == "ci95":
        summary = grouped.agg(["std", "count"]).reindex(mean.index).fillna(0.0)
        sem = summary["std"] / np.sqrt(summary["count"].replace(0, np.nan))
        sem = sem.fillna(0.0)
        margin = 1.96 * sem
        lower = mean - margin
        upper = mean + margin
    else:
        raise ValueError(f"Unknown band mode: {band_mode}")

    return mean, lower, upper


def _ci95_summary(values: pd.Series):
    cleaned = values.dropna()
    count = int(cleaned.shape[0])
    if count == 0:
        return np.nan, np.nan, np.nan, 0

    mean = float(cleaned.mean())
    if count == 1:
        return mean, mean, mean, 1

    sem = float(cleaned.std(ddof=1) / np.sqrt(count))
    margin = 1.96 * sem
    return mean, mean - margin, mean + margin, count


def _print_final_slot_ci95(total_metrics_df, total_profits_df, names, normalized=False, label=None):
    x_col = "slot" if not normalized else "normalized_slots"

    if label is not None:
        print(f"\n[final-slot-ci95] {label}")
    else:
        print("\n[final-slot-ci95]")
    print("configuration\tmetric\tmean\tci95_lower\tci95_upper\tn")

    metric_sources = [
        ("gini", total_metrics_df),
        ("liveness", total_metrics_df),
        ("hhi", total_metrics_df),
        ("cv", total_profits_df),
    ]

    for name in names:
        for metric_name, source_df in metric_sources:
            subset = source_df[source_df["name"] == name]
            if subset.empty:
                mean = lower = upper = np.nan
                count = 0
            else:
                final_x = subset[x_col].max()
                final_values = subset.loc[subset[x_col] == final_x, metric_name]
                mean, lower, upper, count = _ci95_summary(final_values)

            print(f"{name}\t{metric_name}\t{mean:.6f}\t{lower:.6f}\t{upper:.6f}\t{count}")


def _plot_mean_with_uncertainty(
    ax: plt.Axes,
    data_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    names,
    ylabel: str,
    band_mode: str,
    legend: bool = True,
    xlabel: str = "Slot",
):
    sns.set_style("whitegrid")
    palette = sns.color_palette("deep", n_colors=len(names))
    line_styles = {
        name: "-" if idx < max(1, len(names) // 2) else (0, (2, 2))
        for idx, name in enumerate(names)
    }
    marker_cycle = ["o", "X", "D", "^", "v", "P", "s", "*"]

    handles = []
    for idx, name in enumerate(names):
        subset = data_df[data_df["name"] == name]
        grouped = subset.groupby(x_col)[y_col]
        mean, lower, upper = _compute_uncertainty_bounds(grouped, band_mode)
        line, = ax.plot(
            mean.index,
            mean.values,
            color=palette[idx],
            linewidth=8.0,
            linestyle=line_styles[name],
            marker=marker_cycle[idx % len(marker_cycle)],
            markevery=2000,
            markersize=20,
            label=name,
        )
        ax.fill_between(
            mean.index,
            lower.values,
            upper.values,
            color=palette[idx],
            alpha=0.18,
            linewidth=0,
        )
        handles.append(line)

    ax.set_xlabel(xlabel, fontsize=40)
    ax.set_ylabel(ylabel, fontsize=40)
    if xlabel == "Slot":
        ax.set_xlim(0, data_df[x_col].max() + 1)
    if xlabel is not None:
        ax.tick_params(axis="x", labelsize=36)
    else:
        ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis="y", labelsize=36)

    if not legend and ax.get_legend() is not None:
        ax.get_legend().remove()

    return handles


def plot_comparision_with_uncertainty(
    folder_path_groups,
    names,
    output_path,
    band_mode="iqr",
    figsize=(25, 12),
    normalized=False,
    ncol=None,
    columnspacing=None,
    h_offset=0.04,
    show_band_legend=True,
):
    total_metrics_df, total_profits_df = _load_comparison_run_data(
        folder_path_groups,
        names,
        normalized=normalized,
    )

    if ncol is None:
        ncol = 4 if show_band_legend else len(names)

    if columnspacing is None:
        columnspacing = 2.0

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharey=False, sharex=True, dpi=300)
    axes = axes.flatten()
    x_col = "slot" if not normalized else "normalized_slots"
    x_label = "Slot" if not normalized else "Relative Progress"

    _print_final_slot_ci95(
        total_metrics_df,
        total_profits_df,
        names,
        normalized=normalized,
        label=Path(output_path).name,
    )

    for idx, y, y_label in zip(
        range(3),
        ["gini", "liveness", "hhi"],
        [r"$\mathrm{Gini}_{\mathrm{g}}$", r"$\mathrm{LC}_{\mathrm{g}}$", r"$\mathrm{HHI}_{\mathrm{g}}$"],
    ):
        _plot_mean_with_uncertainty(
            axes[idx],
            total_metrics_df,
            x_col,
            y,
            names,
            y_label,
            band_mode,
            legend=False,
            xlabel=x_label if idx == 2 else None,
        )

    handles = _plot_mean_with_uncertainty(
        axes[-1],
        total_profits_df,
        x_col,
        "cv",
        names,
        r"$\mathrm{CV}_{\mathrm{g}}$",
        band_mode,
        legend=True,
        xlabel=x_label,
    )

    if axes[-1].get_legend() is not None:
        axes[-1].get_legend().remove()

    plt.subplots_adjust(hspace=0.1, wspace=0.3)
    legend_handles = list(handles)
    legend_labels = list(names)
    if show_band_legend:
        band_suffix = "95% CI" if band_mode == "ci95" else "IQR"
        legend_handles.extend(
            [
                Patch(facecolor=handle.get_color(), alpha=0.18, edgecolor="none")
                for handle in handles
            ]
        )
        legend_labels.extend(
            [
                name.replace("(Mean)", f"({band_suffix})") if "(Mean)" in name else f"{name} ({band_suffix})"
                for name in names
            ]
        )

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        fontsize=40,
        ncol=ncol,
        title=None,
        bbox_to_anchor=(0.5, 1 + h_offset),
        framealpha=0,
        facecolor="none",
        edgecolor="none",
        columnspacing=columnspacing,
    )
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


@task
def plot_baseline(c, seed=None, latency_std_dev_ratio=None, max_runs=20):
    """Generate baseline plots."""
    folder_path_groups = [
        _resolve_run_group(
            os.path.join(OUTPUT_DIR, "baseline", "SSP", "validators_1000_slots_10000_cost_0.002"),
            seed=seed,
            latency_std_dev_ratio=latency_std_dev_ratio,
            max_runs=max_runs,
        ),
        _resolve_run_group(
            os.path.join(OUTPUT_DIR, "baseline", "MSP", "validators_1000_slots_10000_cost_0.002"),
            seed=seed,
            latency_std_dev_ratio=latency_std_dev_ratio,
            max_runs=max_runs,
        ),
    ]

    names = [
        "External",
        "Local",
    ]

    ylabels = [
        "Validator Distribution (%)",
        "Validator Distribution (%)",
    ]

    continent_comparision_output_path = _figure_output_path("continent_comparision_baseline.pdf", seed, latency_std_dev_ratio)
    continent_distribution_output_path = _figure_output_path("continent_distribution_baseline.pdf", seed, latency_std_dev_ratio)

    if seed is None:
        plot_comparision_mean(folder_path_groups, names, continent_comparision_output_path, h_offset=0)
        plot_continent_distribution_mean(folder_path_groups, names, ylabels, continent_distribution_output_path, subplots=(1,2))
    else:
        folder_paths = [str(run_group[0]) for run_group in folder_path_groups]
        plot_comparision(folder_paths, names, continent_comparision_output_path, h_offset=0)
        plot_continent_distribution(folder_paths, names, ylabels, continent_distribution_output_path, subplots=(1,2))


@task
def plot_baseline_mean_iqr(c, latency_std_dev_ratio=None, max_runs=20):
    """Generate baseline mean curves with interquartile-range uncertainty bands across seeded runs."""
    folder_path_groups = [
        _collect_seeded_run_dirs(
            os.path.join(OUTPUT_DIR, "baseline", "SSP", "validators_1000_slots_10000_cost_0.002"),
            latency_std_dev_ratio=latency_std_dev_ratio,
            max_runs=max_runs,
        ),
        _collect_seeded_run_dirs(
            os.path.join(OUTPUT_DIR, "baseline", "MSP", "validators_1000_slots_10000_cost_0.002"),
            latency_std_dev_ratio=latency_std_dev_ratio,
            max_runs=max_runs,
        ),
    ]

    output_path = _figure_output_path(
        "continent_comparision_baseline_mean_iqr.pdf",
        latency_std_dev_ratio=latency_std_dev_ratio,
    )

    plot_comparision_with_uncertainty(
        folder_path_groups,
        ["External (Mean)", "Local (Mean)"],
        output_path,
        band_mode="iqr",
        figsize=(25, 12),
        ncol=4,
        columnspacing=0.75,
        h_offset=0,
    )


@task
def plot_baseline_mean_ci(c, latency_std_dev_ratio=None, max_runs=20):
    """Generate baseline mean curves with 95% confidence-interval uncertainty bands across seeded runs."""
    folder_path_groups = [
        _collect_seeded_run_dirs(
            os.path.join(OUTPUT_DIR, "baseline", "SSP", "validators_1000_slots_10000_cost_0.002"),
            latency_std_dev_ratio=latency_std_dev_ratio,
            max_runs=max_runs,
        ),
        _collect_seeded_run_dirs(
            os.path.join(OUTPUT_DIR, "baseline", "MSP", "validators_1000_slots_10000_cost_0.002"),
            latency_std_dev_ratio=latency_std_dev_ratio,
            max_runs=max_runs,
        ),
    ]

    output_path = _figure_output_path(
        "continent_comparision_baseline_mean_ci95.pdf",
        latency_std_dev_ratio=latency_std_dev_ratio,
    )

    plot_comparision_with_uncertainty(
        folder_path_groups,
        ["External (Mean)", "Local (Mean)"],
        output_path,
        band_mode="ci95",
        figsize=(25, 12),
        ncol=4,
        columnspacing=0.75,
        h_offset=0,
    )


@task
def plot_baseline_latency_std_dev_variation(c, seed=None):
    """Generate one baseline comparison plot for External and Local latency std-dev variations."""
    ratios = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    comparison_folder_paths = []
    comparison_names = []
    dashes = []

    for model, label, dash_pattern in [
        ("SSP", "External", (None, None)),
        ("MSP", "Local", (2, 2)),
    ]:
        comparison_folder_paths.extend(
            [_baseline_variation_path(model, ratio, seed) for ratio in ratios]
        )
        comparison_names.extend(
            [rf"$\sigma={ratio:.1f}$ ({label})" for ratio in ratios]
        )
        dashes.extend([dash_pattern] * len(ratios))

    continent_comparision_output_path = _figure_output_path(
        "continent_comparision_baseline_latency_std_dev_variation.pdf",
        seed,
    )
    plot_comparision(
        comparison_folder_paths,
        comparison_names,
        continent_comparision_output_path,
        figsize=(25, 13),
        desired_order=comparison_names,
        ncol=2,
        columnspacing=8,
        h_offset=0.34,
        dashes=dashes,
    )


@task
def plot_cost(c, seed=None, latency_std_dev_ratio=None, max_runs=20):
    """Generate cost plots."""
    folder_path_groups = []
    names = []

    label_map = {
        "SSP" : "External",
        "MSP" : "Local",
    }


    for paradigm in ["SSP", "MSP"]:
        for cost in ["0.0", "0.001", "0.002", "0.003"]:
            folder_path_groups.append(
                _resolve_run_group(
                    os.path.join(OUTPUT_DIR, "baseline", paradigm, f"validators_1000_slots_10000_cost_{cost}"),
                    seed=seed,
                    latency_std_dev_ratio=latency_std_dev_ratio,
                    max_runs=max_runs,
                )
            )
            names.append(f"$c = {cost}$ ({label_map[paradigm]})")


    continent_comparision_output_path = _figure_output_path("continent_comparision_cost.pdf", seed, latency_std_dev_ratio)

    plot_comparision_mean(folder_path_groups, names, continent_comparision_output_path)


@task
def plot_heterogeneous_information_sources(c, seed=None, latency_std_dev_ratio=None, max_runs=20):
    """
    Generate hetero-info plots.
    """
    folder_path_groups = [
        _resolve_run_group(os.path.join(OUTPUT_DIR, "baseline", "SSP", "validators_1000_slots_10000_cost_0.002"), seed=seed, latency_std_dev_ratio=latency_std_dev_ratio, max_runs=max_runs),
        _resolve_run_group(os.path.join(OUTPUT_DIR, "baseline", "MSP", "validators_1000_slots_10000_cost_0.002"), seed=seed, latency_std_dev_ratio=latency_std_dev_ratio, max_runs=max_runs),
        _resolve_run_group(os.path.join(OUTPUT_DIR, "hetero_info", "SSP", "validators_1000_slots_10000_cost_0.002_latency_latency-aligned"), seed=seed, latency_std_dev_ratio=latency_std_dev_ratio, max_runs=max_runs),
        _resolve_run_group(os.path.join(OUTPUT_DIR, "hetero_info", "MSP", "validators_1000_slots_10000_cost_0.002_latency_latency-aligned"), seed=seed, latency_std_dev_ratio=latency_std_dev_ratio, max_runs=max_runs),
        _resolve_run_group(os.path.join(OUTPUT_DIR, "hetero_info", "SSP", "validators_1000_slots_10000_cost_0.002_latency_latency-misaligned"), seed=seed, latency_std_dev_ratio=latency_std_dev_ratio, max_runs=max_runs),
        _resolve_run_group(os.path.join(OUTPUT_DIR, "hetero_info", "MSP", "validators_1000_slots_10000_cost_0.002_latency_latency-misaligned"), seed=seed, latency_std_dev_ratio=latency_std_dev_ratio, max_runs=max_runs),
    ]

    names = [
        "baseline (External)",
        "baseline (Local)",
        "latency-aligned (External)",
        "latency-aligned (Local)",
        "latency-misaligned (External)",
        "latency-misaligned (Local)",
    ]

    order = [
        "baseline (External)",
        "latency-aligned (External)",
        "latency-misaligned (External)",
        "baseline (Local)",
        "latency-aligned (Local)",
        "latency-misaligned (Local)",
    ]

    continent_comparision_output_path = _figure_output_path("continent_comparision_hetero_info.pdf", seed, latency_std_dev_ratio)
    plot_comparision_mean(folder_path_groups, names, continent_comparision_output_path, figsize=(25, 13), desired_order=order, ncol=2, columnspacing=2.0, h_offset=0.09)


@task
def plot_heterogeneous_validators(c, seed=None, latency_std_dev_ratio=None, max_runs=20):
    """Generate hetero-validator plots."""
    folder_path_groups = [
        _resolve_run_group(os.path.join(OUTPUT_DIR, "baseline", "SSP", "validators_1000_slots_10000_cost_0.002"), seed=seed, latency_std_dev_ratio=latency_std_dev_ratio, max_runs=max_runs),
        _resolve_run_group(os.path.join(OUTPUT_DIR, "baseline", "MSP", "validators_1000_slots_10000_cost_0.002"), seed=seed, latency_std_dev_ratio=latency_std_dev_ratio, max_runs=max_runs),
        _resolve_run_group(os.path.join(OUTPUT_DIR, "hetero_validators", "SSP", "slots_10000_cost_0.002_validators_heterogeneous"), seed=seed, latency_std_dev_ratio=latency_std_dev_ratio, max_runs=max_runs),
        _resolve_run_group(os.path.join(OUTPUT_DIR, "hetero_validators", "MSP", "slots_10000_cost_0.002_validators_heterogeneous"), seed=seed, latency_std_dev_ratio=latency_std_dev_ratio, max_runs=max_runs),
    ]

    names = [
        "baseline (External)",
        "baseline (Local)",
        "heterogeneous validators (External)",
        "heterogeneous validators (Local)",
    ]

    order = [
        "baseline (External)",
        "heterogeneous validators (External)",
        "baseline (Local)",
        "heterogeneous validators (Local)",
    ]

    continent_comparision_output_path = _figure_output_path("continent_comparision_hetero_validator.pdf", seed, latency_std_dev_ratio)
    plot_comparision_mean(folder_path_groups, names, continent_comparision_output_path, desired_order=order, ncol=2, columnspacing=2.0)


@task
def plot_hetero_both(c, seed=None, latency_std_dev_ratio=None, max_runs=20):
    """Generate hetero-both plots."""
    folder_path_groups = [
        _resolve_run_group(os.path.join(OUTPUT_DIR, "baseline", "SSP", "validators_1000_slots_10000_cost_0.002"), seed=seed, latency_std_dev_ratio=latency_std_dev_ratio, max_runs=max_runs),
        _resolve_run_group(os.path.join(OUTPUT_DIR, "baseline", "MSP", "validators_1000_slots_10000_cost_0.002"), seed=seed, latency_std_dev_ratio=latency_std_dev_ratio, max_runs=max_runs),
        _resolve_run_group(os.path.join(OUTPUT_DIR, "hetero_both", "SSP", "validators_heterogeneous_slots_10000_cost_0.002_latency_latency-aligned"), seed=seed, latency_std_dev_ratio=latency_std_dev_ratio, max_runs=max_runs),
        _resolve_run_group(os.path.join(OUTPUT_DIR, "hetero_both", "MSP", "validators_heterogeneous_slots_10000_cost_0.002_latency_latency-aligned"), seed=seed, latency_std_dev_ratio=latency_std_dev_ratio, max_runs=max_runs),
        _resolve_run_group(os.path.join(OUTPUT_DIR, "hetero_both", "SSP", "validators_heterogeneous_slots_10000_cost_0.002_latency_latency-misaligned"), seed=seed, latency_std_dev_ratio=latency_std_dev_ratio, max_runs=max_runs),
        _resolve_run_group(os.path.join(OUTPUT_DIR, "hetero_both", "MSP", "validators_heterogeneous_slots_10000_cost_0.002_latency_latency-misaligned"), seed=seed, latency_std_dev_ratio=latency_std_dev_ratio, max_runs=max_runs),
    ]

    names = [
        "baseline (External)",
        "baseline (Local)",
        "latency-aligned (External)",
        "latency-aligned (Local)",
        "latency-misaligned (External)",
        "latency-misaligned (Local)",
    ]

    order = [
        "baseline (External)",
        "latency-aligned (External)",
        "latency-misaligned (External)",
        "baseline (Local)",
        "latency-aligned (Local)",
        "latency-misaligned (Local)",
    ]

    distribution_folder_path_groups = [
        folder_path_groups[2],
        folder_path_groups[4],
        folder_path_groups[3],
        folder_path_groups[5],
    ]

    xlabels = [
        "latency-aligned (External)",
        "latency-misaligned (External)",
        "latency-aligned (Local)",
        "latency-misaligned (Local)",
    ]

    ylabels = [
        "Validator Distribution (%)",
        "Validator Distribution (%)",
        "Validator Distribution (%)",
        "Validator Distribution (%)",
    ]

    continent_comparision_output_path = _figure_output_path("continent_comparision_hetero_both.pdf", seed, latency_std_dev_ratio)
    continent_distribution_output_path = _figure_output_path("continent_distribution_hetero_both.pdf", seed, latency_std_dev_ratio)
    plot_comparision_mean(folder_path_groups, names, continent_comparision_output_path, figsize=(25, 13), desired_order=order, ncol=2, columnspacing=2.0, h_offset=0.09)
    plot_continent_distribution_mean(distribution_folder_path_groups, xlabels, ylabels, continent_distribution_output_path)


@task
def plot_different_gammas(c, seed=None, latency_std_dev_ratio=None, max_runs=20):
    """Generate gamma plots."""
    folder_path_groups = [
        _resolve_run_group(os.path.join(OUTPUT_DIR, "different_gammas", "SSP", "validators_1000_slots_10000_cost_0.002_gamma_0.3333"), seed=seed, latency_std_dev_ratio=latency_std_dev_ratio, max_runs=max_runs),
        _resolve_run_group(os.path.join(OUTPUT_DIR, "different_gammas", "SSP", "validators_1000_slots_10000_cost_0.002_gamma_0.5"), seed=seed, latency_std_dev_ratio=latency_std_dev_ratio, max_runs=max_runs),
        _resolve_run_group(os.path.join(OUTPUT_DIR, "baseline", "SSP", "validators_1000_slots_10000_cost_0.002"), seed=seed, latency_std_dev_ratio=latency_std_dev_ratio, max_runs=max_runs),
        _resolve_run_group(os.path.join(OUTPUT_DIR, "different_gammas", "SSP", "validators_1000_slots_10000_cost_0.002_gamma_0.8"), seed=seed, latency_std_dev_ratio=latency_std_dev_ratio, max_runs=max_runs),
        _resolve_run_group(os.path.join(OUTPUT_DIR, "different_gammas", "MSP", "validators_1000_slots_10000_cost_0.002_gamma_0.3333"), seed=seed, latency_std_dev_ratio=latency_std_dev_ratio, max_runs=max_runs),
        _resolve_run_group(os.path.join(OUTPUT_DIR, "different_gammas", "MSP", "validators_1000_slots_10000_cost_0.002_gamma_0.5"), seed=seed, latency_std_dev_ratio=latency_std_dev_ratio, max_runs=max_runs),
        _resolve_run_group(os.path.join(OUTPUT_DIR, "baseline", "MSP", "validators_1000_slots_10000_cost_0.002"), seed=seed, latency_std_dev_ratio=latency_std_dev_ratio, max_runs=max_runs),
        _resolve_run_group(os.path.join(OUTPUT_DIR, "different_gammas", "MSP", "validators_1000_slots_10000_cost_0.002_gamma_0.8"), seed=seed, latency_std_dev_ratio=latency_std_dev_ratio, max_runs=max_runs),
    ]

    names = [r"$\gamma=1/3$ (External)", r"$\gamma=1/2$ (External)", r"$\gamma=2/3$ (External)", r"$\gamma=4/5$ (External)",
             r"$\gamma=1/3$ (Local)", r"$\gamma=1/2$ (Local)", r"$\gamma=2/3$ (Local)", r"$\gamma=4/5$ (Local)"]

    continent_comparision_output_path = _figure_output_path("continent_comparision_gamma.pdf", seed, latency_std_dev_ratio)
    plot_comparision_mean(folder_path_groups, names, continent_comparision_output_path)


@task
def plot_eip7782(c, seed=None, latency_std_dev_ratio=None, max_runs=20):
    """Generate EIP-7782 plots."""
    folder_path_groups = [
        _resolve_run_group(os.path.join(OUTPUT_DIR, "baseline", "SSP", "validators_1000_slots_10000_cost_0.002"), seed=seed, latency_std_dev_ratio=latency_std_dev_ratio, max_runs=max_runs),
        _resolve_run_group(os.path.join(OUTPUT_DIR, "eip7782", "SSP", "validators_1000_slots_10000_cost_0.002_delta_6000_cutoff_3000"), seed=seed, latency_std_dev_ratio=latency_std_dev_ratio, max_runs=max_runs),
        _resolve_run_group(os.path.join(OUTPUT_DIR, "baseline", "MSP", "validators_1000_slots_10000_cost_0.002"), seed=seed, latency_std_dev_ratio=latency_std_dev_ratio, max_runs=max_runs),
        _resolve_run_group(os.path.join(OUTPUT_DIR, "eip7782", "MSP", "validators_1000_slots_10000_cost_0.002_delta_6000_cutoff_3000"), seed=seed, latency_std_dev_ratio=latency_std_dev_ratio, max_runs=max_runs),
    ]

    names = [
        r"$\Delta=12s$ (External)",
        r"$\Delta=6s$ (External)",
        r"$\Delta=12s$ (Local)",
        r"$\Delta=6s$ (Local)"
    ]

    continent_comparision_output_path = _figure_output_path("continent_comparision_eip7782.pdf", seed, latency_std_dev_ratio)

    plot_comparision_mean(folder_path_groups, names, continent_comparision_output_path)


@task
def plot_different_scale(c, seed=None, latency_std_dev_ratio=None):
    """Generate different scale plots."""

    folder_paths = [
        _with_run_suffix(os.path.join(OUTPUT_DIR, "baseline", "SSP", "validators_100_slots_1000_cost_0.0"), seed, latency_std_dev_ratio),
        _with_run_suffix(os.path.join(OUTPUT_DIR, "baseline", "SSP", "validators_1000_slots_10000_cost_0.0"), seed, latency_std_dev_ratio),
        _with_run_suffix(os.path.join(OUTPUT_DIR, "baseline", "SSP", "validators_10000_slots_100000_cost_0.0"), seed, latency_std_dev_ratio),
        _with_run_suffix(os.path.join(OUTPUT_DIR, "baseline", "MSP", "validators_100_slots_1000_cost_0.0"), seed, latency_std_dev_ratio),
        _with_run_suffix(os.path.join(OUTPUT_DIR, "baseline", "MSP", "validators_1000_slots_10000_cost_0.0"), seed, latency_std_dev_ratio),
        _with_run_suffix(os.path.join(OUTPUT_DIR, "baseline", "MSP", "validators_10000_slots_100000_cost_0.0"), seed, latency_std_dev_ratio),
    ]

    names = [
        r"$|\mathcal{V}| = 100$ (External)",
        r"$|\mathcal{V}| = 1,000$ (External)",
        r"$|\mathcal{V}| = 10,000$ (External)",
        r"$|\mathcal{V}| = 100$ (Local)",
        r"$|\mathcal{V}| = 1,000$ (Local)",
        r"$|\mathcal{V}| = 10,000$ (Local)",
    ]

    continent_comparision_output_path = _figure_output_path("continent_comparision_different_scale.pdf", seed, latency_std_dev_ratio)

    plot_comparision(folder_paths, names, continent_comparision_output_path, figsize=(25, 13), normalized=True, ncol=2, columnspacing=2.0, h_offset=0.09)
