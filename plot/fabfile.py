# fabfile.py
from fabric import task
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import os
import re


from collections import defaultdict, Counter
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


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


def plot_continent_distribution(folder_paths, xlabels, ylabels, output_path):
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(34, 15), dpi=300, sharey=True)

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
    fig.legend(handles, labels, title=None, fontsize=42, ncol=7, loc='upper center', bbox_to_anchor=(0.5, 1.08), framealpha=0, facecolor="none", edgecolor="none", columnspacing=0.5)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')


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


def compute_metrics(folder_path: Path) -> Dict[str, Any]:
    region_counts_per_slot = _load_json_if_exists(
        folder_path / "region_counter_per_slot.json"
    )
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


def single_line(ax: plt.Axes, data_df: pd.DataFrame, x_col: str, y_col: str, hue: str, ylabel: str, legend: bool = True, xlabel: str = "Slot"):
    count = data_df[hue].nunique() // 2
    
    sns.lineplot(
        data=data_df,
        x=x_col,
        y=y_col,
        hue=hue,
        style=hue,
        dashes=[(None, None)] * count + [(2,2)] * count,
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


def plot_comparision(folder_paths, names, output_path, normalized=False):
    total_metrcis = []
    total_profits = []

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
    
    import IPython
    IPython.embed(colors="neutral")

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2,2, figsize=(25, 12), sharey=False, sharex=True, dpi=300)
    axes = axes.flatten()
    for idx, y, y_label in zip(range(3), ["gini", "liveness", "hhi"], ["Gini$_{g}$",  "LC$_{g}$", "HHI$_{g}$"]):
        single_line(
            axes[idx],
            total_metrics_df,
            "slot" if not normalized else "normalized_slots",
            y,
            "name",
            y_label,
            False,
            ("Slot" if not normalized else "Relative Progress") if idx == 2 else None
        )

    # cv    
    single_line(
        axes[-1],
        total_profits_df,
        "slot" if not normalized else "normalized_slots",
        "cv",
        "name",
        "CV$_{g}$",
        True,
        "Slot" if not normalized else "Relative Progress"
    )

    plt.subplots_adjust(hspace=0.1, wspace=0.3)
    handles, labels = axes[-1].get_legend_handles_labels()
    axes[-1].legend_.remove()

    fig.legend(handles, labels, loc="upper center", fontsize=40, ncol=len(names)//2, title=None, bbox_to_anchor=(0.5, 1.04), framealpha=0, facecolor="none", edgecolor="none", columnspacing=round(10.5-1.25*len(names), 1))
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    
@task
def plot_baseline(c):
    """Generate baseline plots."""
    folder_paths = [
        os.path.join(OUTPUT_DIR, "baseline", "SSP", "validators_1000_slots_10000_cost_0.0"),
        os.path.join(OUTPUT_DIR, "baseline", "SSP", "validators_1000_slots_10000_cost_0.002"),
        os.path.join(OUTPUT_DIR, "baseline", "MSP", "validators_1000_slots_10000_cost_0.0"),
        os.path.join(OUTPUT_DIR, "baseline", "MSP", "validators_1000_slots_10000_cost_0.002"),
    ]

    names = xlabels = [
        "$c = 0$ (SSP)",
        "$c = 0.002$ (SSP)",
        "$c = 0$ (MSP)",
        "$c = 0.002$ (MSP)",
    ]

    ylabels = [
        "Validator Distribution (%)",
        "Validator Distribution (%)",
        "Validator Distribution (%)",
        "Validator Distribution (%)",
    ]

    continent_comparision_output_path = os.path.join(FIGURE_DIR, "continent_comparision_baseline.pdf")
    continent_distribution_output_path = os.path.join(FIGURE_DIR, "continent_distribution_baseline.pdf")

    plot_comparision(folder_paths, names, continent_comparision_output_path)
    plot_continent_distribution(folder_paths, xlabels, ylabels, continent_distribution_output_path)
    

@task
def plot_different_scale(c):
    """Generate different scale plots."""

    folder_paths = [
        os.path.join("/home/senyang/geographical-decentralization-simulation", "baseline", "SSP", "validators_100_slots_1000_cost_0.0"),
        os.path.join("/home/senyang/geographical-decentralization-simulation/output2", "baseline", "SSP", "validators_1000_slots_10000_cost_0.0"),
        os.path.join("/home/senyang/geographical-decentralization-simulation", "baseline", "SSP", "validators_10000_slots_100000_cost_0.0"),
        os.path.join("/home/senyang/geographical-decentralization-simulation", "baseline", "MSP", "validators_100_slots_1000_cost_0.0"),
        os.path.join("/home/senyang/geographical-decentralization-simulation/output2", "baseline", "MSP", "validators_1000_slots_10000_cost_0.0"),
        os.path.join("/home/senyang/geographical-decentralization-simulation", "baseline", "MSP", "validators_10000_slots_100000_cost_0.0"),
    ]

    names = [
        r"$|\mathcal{V}| = 100$ (SSP)",
        r"$|\mathcal{V}| = 1,000$ (SSP)",
        r"$|\mathcal{V}| = 10,000$ (SSP)",
        r"$|\mathcal{V}| = 100$ (MSP)",
        r"$|\mathcal{V}| = 1,000$ (MSP)",
        r"$|\mathcal{V}| = 10,000$ (MSP)",
    ]

    continent_comparision_output_path = os.path.join(FIGURE_DIR, "continent_comparision_different_scale.pdf")

    plot_comparision(folder_paths, names, continent_comparision_output_path, normalized=True)


@task
def plot_cost(c):
    """Generate cost plots."""
    folder_paths = []
    names = []
    
    for paradigm in ["SSP", "MSP"]:
        for cost in ["0.0", "0.001", "0.002", "0.003"]:
            folder_paths.append(
                os.path.join(OUTPUT_DIR, "baseline", paradigm, f"validators_1000_slots_10000_cost_{cost}")
            )
            names.append(f"$c = {cost}$ ({paradigm})")


    continent_comparision_output_path = os.path.join(FIGURE_DIR, "continent_comparision_cost.pdf")

    plot_comparision(folder_paths, names, continent_comparision_output_path)


@task
def plot_heterogeneous_information_sources(c):
    """
    Generate hetero-info plots.
    """
    folder_paths = [
        os.path.join(OUTPUT_DIR, "hetero_info", "SSP", "validators_1000_slots_10000_cost_0.002_latency_latency-aligned"),
        os.path.join(OUTPUT_DIR, "hetero_info", "SSP", "validators_1000_slots_10000_cost_0.002_latency_latency-misaligned"),
        os.path.join(OUTPUT_DIR, "hetero_info", "MSP", "validators_1000_slots_10000_cost_0.002_latency_latency-aligned"),
        os.path.join(OUTPUT_DIR, "hetero_info", "MSP", "validators_1000_slots_10000_cost_0.002_latency_latency-misaligned"),
    ]

    names = [
        "latency-aligned (SSP)",
        "latency-misaligned (SSP)",
        "latency-aligned (MSP)",
        "latency-misaligned (MSP)",
    ]

    continent_comparision_output_path = os.path.join(FIGURE_DIR, "continent_comparision_hetero_info.pdf")
    plot_comparision(folder_paths, names, continent_comparision_output_path)


@task
def plot_heterogeneous_validators(c):
    """Generate hetero-validator plots."""
    folder_paths = [
        os.path.join(OUTPUT_DIR, "hetero_validators", "SSP", "slots_10000_cost_0.0_validators_heterogeneous"),
        os.path.join(OUTPUT_DIR, "hetero_validators", "SSP", "slots_10000_cost_0.002_validators_heterogeneous"),
        os.path.join(OUTPUT_DIR, "hetero_validators", "MSP", "slots_10000_cost_0.0_validators_heterogeneous"),
        os.path.join(OUTPUT_DIR, "hetero_validators", "MSP", "slots_10000_cost_0.002_validators_heterogeneous"),
    ]

    names = [
        "$c = 0$ (SSP)",
        "$c = 0.002$ (SSP)",
        "$c = 0$ (MSP)",
        "$c = 0.002$ (MSP)",
    ]

    continent_comparision_output_path = os.path.join(FIGURE_DIR, "continent_comparision_hetero_validator.pdf")
    plot_comparision(folder_paths, names, continent_comparision_output_path)


@task
def plot_hetero_both(c):
    """Generate hetero-both plots."""
    folder_paths = [
        os.path.join(OUTPUT_DIR, "hetero_both", "SSP", "validators_heterogeneous_slots_10000_cost_0.002_latency_latency-aligned"),
        os.path.join(OUTPUT_DIR, "hetero_both", "SSP", "validators_heterogeneous_slots_10000_cost_0.002_latency_latency-misaligned"),
        os.path.join(OUTPUT_DIR, "hetero_both", "MSP", "validators_heterogeneous_slots_10000_cost_0.002_latency_latency-aligned"),
        os.path.join(OUTPUT_DIR, "hetero_both", "MSP", "validators_heterogeneous_slots_10000_cost_0.002_latency_latency-misaligned"),
    ]

    names = xlabels = [
        "latency-aligned (SSP)",
        "latency-misaligned (SSP)",
        "latency-aligned (MSP)",
        "latency-misaligned (MSP)",
    ]

    ylabels = [
        "Validator Distribution (%)",
        "Validator Distribution (%)",
        "Validator Distribution (%)",
        "Validator Distribution (%)",
    ]

    continent_comparision_output_path = os.path.join(FIGURE_DIR, "continent_comparision_hetero_both.pdf")
    continent_distribution_output_path = os.path.join(FIGURE_DIR, "continent_distribution_hetero_both.pdf")
    plot_comparision(folder_paths, names, continent_comparision_output_path)
    plot_continent_distribution(folder_paths, xlabels, ylabels, continent_distribution_output_path)


@task
def plot_different_gammas(c):
    """Generate gamma plots."""
    folder_paths = [
        os.path.join(OUTPUT_DIR, "different_gammas", "SSP", "validators_1000_slots_10000_cost_0.002_gamma_0.3333"),
        os.path.join(OUTPUT_DIR, "different_gammas", "SSP", "validators_1000_slots_10000_cost_0.002_gamma_0.5"),
        os.path.join(OUTPUT_DIR, "baseline", "SSP", "validators_1000_slots_10000_cost_0.002"),
        os.path.join(OUTPUT_DIR, "different_gammas", "SSP", "validators_1000_slots_10000_cost_0.002_gamma_0.8"),
        os.path.join(OUTPUT_DIR, "different_gammas", "MSP", "validators_1000_slots_10000_cost_0.002_gamma_0.3333"),
        os.path.join(OUTPUT_DIR, "different_gammas", "MSP", "validators_1000_slots_10000_cost_0.002_gamma_0.5"),
        os.path.join(OUTPUT_DIR, "baseline", "MSP", "validators_1000_slots_10000_cost_0.002"),
        os.path.join(OUTPUT_DIR, "different_gammas", "MSP", "validators_1000_slots_10000_cost_0.002_gamma_0.8"),
    ]

    names = [r"$\gamma=1/3$ (SSP)", r"$\gamma=1/2$ (SSP)", r"$\gamma=2/3$ (SSP)", r"$\gamma=4/5$ (SSP)", r"$\gamma=1/3$ (MSP)", r"$\gamma=1/2$ (MSP)", r"$\gamma=2/3$ (MSP)", r"$\gamma=4/5$ (MSP)"]

    continent_comparision_output_path = os.path.join(FIGURE_DIR, "continent_comparision_gamma.pdf")
    plot_comparision(folder_paths, names, continent_comparision_output_path)


@task
def plot_eip7782(c):
    """Generate EIP-7782 plots."""
    folder_paths = [
        os.path.join(OUTPUT_DIR, "baseline", "SSP", "validators_1000_slots_10000_cost_0.002"),
        os.path.join(OUTPUT_DIR, "eip7782", "SSP", "validators_1000_slots_10000_cost_0.002_delta_6000_cutoff_3000"),
        os.path.join(OUTPUT_DIR, "baseline", "MSP", "validators_1000_slots_10000_cost_0.002"),
        os.path.join(OUTPUT_DIR, "eip7782", "MSP", "validators_1000_slots_10000_cost_0.002_delta_6000_cutoff_3000"),
    ]

    names = [
        r"$\Delta=12s$ (SSP)",
        r"$\Delta=6s$ (SSP)",
        r"$\Delta=12s$ (MSP)",
        r"$\Delta=6s$ (MSP)"
    ]

    continent_comparision_output_path = os.path.join(FIGURE_DIR, "continent_comparision_eip7782.pdf")

    plot_comparision(folder_paths, names, continent_comparision_output_path)
