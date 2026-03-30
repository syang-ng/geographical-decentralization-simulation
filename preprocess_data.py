import argparse
import json
import math
import pandas as pd
import re
from collections import Counter, defaultdict
from distribution import *
from measure import *

from multiprocessing import Pool, cpu_count
from tqdm import tqdm

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



class SphericalSpace:
    """
    Sample points on (or near) the unit sphere.
    distance() returns geodesic distance (great-circle distance).
    """
    def distance(self, p1, p2):
        """
        Calculates the geodesic distance between two points on a unit sphere.
        Distance = arc length = arccos(dot(p1,p2)).
        """
        dotp = p1[0] * p2[0] + p1[1] * p2[1] + p1[2] * p2[2]
        # Numerical safety clamp for dot product to be within [-1, 1] due to floating point inaccuracies
        dotp = max(-1.0, min(1.0, dotp))
        return math.acos(dotp)


    def get_area(self):
        """Returns the surface area of a unit sphere."""
        return 4 * np.pi


def init_distance_matrix(positions, space):
    """
    Build the initial distance matrix for all node pairs.
    Returns a 2D list (or NumPy array) of shape (n, n).
    """
    n = len(positions)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = space.distance(positions[i], positions[j])
            dist_matrix[i][j] = d
            dist_matrix[j][i] = d  # Symmetric matrix
    return dist_matrix


space = SphericalSpace()


def load_data(file_path):
    """Load data from JSON file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {file_path} not found.")
        return []


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


def compute_paper_metrics(region_counts_per_slot, region_to_country):
    validator_agent_countries = {}
    validator_agent_continents = {}

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

    initial_num_of_regions = len(region_to_country)
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


def preprocess_slot_counter(validator_agent_regions, region_to_country, region_to_xyz):
    """Convert region data to country data."""
    
    validator_agent_countries = {}
    
    for slot, region_list in validator_agent_regions.items():
        country_counter = defaultdict(int)
        for region, count in region_list:
            country = region_to_country.get(region, "Unknown")
            country_counter[country] += count
        validator_agent_countries[slot] = Counter(country_counter).most_common()
    
    all_slot_data = []
    for slot, region_list in sorted(validator_agent_regions.items(), key=lambda x: int(x[0])):
        country_counter = defaultdict(int)
        slot_data = []

        for region, count in region_list:
            country = region_to_country.get(region, "Unknown")
            country_counter[country] += count
            xyz = region_to_xyz.get(region, (0.0, 0.0, 0.0))
            slot_data.extend([xyz] * count)
    
        validator_agent_countries[slot] = Counter(country_counter).most_common()
        all_slot_data.append(slot_data)
    
    return all_slot_data, validator_agent_countries


def precompute_metrics_per_slot(args):
    i, pts, space = args

    if len(pts) <= 1:
        return i, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    dm = init_distance_matrix(pts, space)
    c = cluster_matrix(dm)
    t = total_distance(dm)
    a = average_nearest_neighbor_distance(dm)
    nni = nearest_neighbor_index_spherical(dm, space)[0]

    # mev = sum(mev_series[i]) if mev_series and i < len(mev_series) else 0.0
    # attest = sum(attest_series[i]) if i < len(attest_series) else 0.0
    # proposal_time = (
    #     sum(x for x in proposal_time_series[i] if x > 0)
    #     if proposal_time_series and i < len(proposal_time_series) and proposal_time_series[i]
    #     else 0.0
    # )

    return i, c, t, a, nni



def precompute_metrics(all_slot_data, mev_series, attest_series, proposal_time_series):
    """Pre-compute all spatial metrics for the entire simulation."""
    granularity = 10
    
    clusters_hist, total_dist_hist, avg_nnd_hist, nni_hist = [], [], [], []
    mev_hist, attest_hist, proposal_time_hist = [], [], []
    
    # Placeholders for last computed values
    last_c = last_t = last_a = last_nni = last_mev = last_attest = last_proposal_time = 0.0

    sample_indices = [
        i for i, pts in enumerate(all_slot_data) if i % granularity == 0 and len(pts) > 1
    ]

    tasks = [
        (i, all_slot_data[i], space) for i in sample_indices
    ]

    print("Starting parallel computation of metrics...")
    with Pool(processes=min(cpu_count() // 2, 10)) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(precompute_metrics_per_slot, tasks),
                total=len(tasks),
                desc="Computing metrics"
            )
        )

    results.sort(key=lambda x: x[0])

    by_i = {
        i: (c, t, a, nni) for (i, c, t, a, nni) in results
    }
    print("Completed parallel computation of metrics.")

    for i, pts in enumerate(all_slot_data):
        if i in by_i:
            last_c, last_t, last_a, last_nni = by_i[i]

        last_mev = sum(mev_series[i]) if mev_series and i < len(mev_series) else 0.0
        last_attest = sum(attest_series[i]) if i < len(attest_series) else 0.0
        last_proposal_time = (
            sum(t for t in proposal_time_series[i] if t > 0)
            if proposal_time_series and i < len(proposal_time_series) and proposal_time_series[i]
            else 0.0
        )
        
        clusters_hist.append(last_c)
        total_dist_hist.append(last_t)
        avg_nnd_hist.append(last_a)
        nni_hist.append(last_nni)
        mev_hist.append(last_mev)
        attest_hist.append(last_attest)
        proposal_time_hist.append(last_proposal_time)
    
    return {
        'clusters': clusters_hist,
        'total_distance': total_dist_hist,
        'avg_nnd': avg_nnd_hist,
        'nni': nni_hist,
        'mev': mev_hist,
        'attestations': attest_hist,
        'proposal_times': proposal_time_hist
    }


def precompute_info_distances_per_slot(args):
    i, pts, info_positions, space = args

    if not pts or not info_positions:
        return i, [0.0] * len(info_positions)

    info_dist_per_slot = [
        sum([space.distance(pt, info_pos) for pt in pts]) / len(pts)
        for info_pos in info_positions
    ]

    return i, info_dist_per_slot


def precompute_info_distances(all_slot_data, info_data):
    """Pre-compute distances to relays for all slots."""
    space = SphericalSpace()

    tasks = [
        (i, all_slot_data[i], info_data[i] if i < len(info_data) else [], space)
        for i in range(len(all_slot_data))
    ]

    print("Starting parallel computation of info distances...")
    with Pool(processes=min(cpu_count() // 2, 10)) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(precompute_info_distances_per_slot, tasks),
                total=len(tasks),
                desc="Computing info distances"
            )
        )
    
    results.sort(key=lambda x: x[0])

    return [info_dist_per_slot for (i, info_dist_per_slot) in results]

    
def make_json_serializable(obj):
    """Convert numpy types and other non-serializable objects to JSON-compatible types."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return [make_json_serializable(v) for v in obj]
    else:
        return obj


def infer_validator_count(validator_agent_regions):
    if not validator_agent_regions:
        return 0

    first_slot = min(validator_agent_regions.keys(), key=lambda value: int(value))
    return sum(int(count) for _, count in validator_agent_regions[first_slot])


def normalize_sources(source_entries, model, label_style="published"):
    suffix = "supplier" if model == "SSP" else "signal"
    normalized = []

    for entry in source_entries:
        if isinstance(entry, dict):
            region = entry.get("gcp_region") or entry.get("region") or entry.get("location")
            name = entry.get("unique_id") or entry.get("name") or region
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            name, region = entry[0], entry[1]
        else:
            name = entry
            region = entry

        region_str = str(region)
        if label_style == "published":
            normalized_name = f"{region_str}-{suffix}"
        else:
            normalized_name = str(name)

        normalized.append((normalized_name, region_str))

    return normalized


def build_preprocessed_payload(
    validator_agent_regions,
    n_slots,
    metrics,
    sources,
    validator_count=None,
    cost=None,
    delta=None,
    cutoff=None,
    gamma=None,
    description=None,
):
    payload = {}
    resolved_validator_count = (
        infer_validator_count(validator_agent_regions)
        if validator_count is None
        else validator_count
    )
    payload["v"] = int(resolved_validator_count)

    if delta is not None:
        payload["delta"] = int(delta)
    if cutoff is not None:
        payload["cutoff"] = int(cutoff)
    if cost is not None:
        payload["cost"] = float(cost)
    if gamma is not None:
        payload["gamma"] = float(gamma)
    if description:
        payload["description"] = description

    payload["n_slots"] = n_slots
    payload["metrics"] = metrics
    payload["sources"] = sources
    payload["slots"] = {int(slot): counter for slot, counter in validator_agent_regions.items()}
    return payload


def preprocess_simulation_data(
    data_dir,
    output_dir,
    model="SSP",
    validator_count=None,
    cost=None,
    delta=None,
    cutoff=None,
    gamma=None,
    description=None,
    source_label_style="published",
):
    """
    Preprocess all simulation data and create optimized files for static loading.
    """
    print("Starting data preprocessing...")
    
    # Load raw data
    print("Loading raw data files...")

    mev_series = load_data(f"{output_dir}/mev_by_slot.json")
    attest_series = load_data(f"{output_dir}/attest_by_slot.json")
    failed_block_proposals = load_data(f"{output_dir}/failed_block_proposals.json")
    proposal_time_series = load_data(f"{output_dir}/proposal_time_by_slot.json")
    validator_agent_regions = load_data(f"{output_dir}/region_counter_per_slot.json")
    profits_df = pd.read_csv(f"{output_dir}/region_profits.csv")
    relay_names = load_data(f"{output_dir}/relay_names.json")
    signal_names = load_data(f"{output_dir}/signal_names.json")

    region_df = pd.read_csv(f"{data_dir}/gcp_regions.csv")
    region_to_country = {}
    region_to_xyz = {}
    for region, city in zip(region_df["Region"], region_df["location"]):
        region_to_country[region] = city.split(",")[-1].strip() if "," in city else city.strip()
    for region, x, y, z in zip(region_df["Region"], region_df["x"], region_df["y"], region_df["z"]):
        region_to_xyz[region] = (x, y, z)

    # Pre-process region and country data
    print("Pre-processing region and country data...")
    all_slot_data, validator_agent_countries = preprocess_slot_counter(validator_agent_regions, region_to_country, region_to_xyz)

    n_slots = len(all_slot_data)
    print(f"Processing {n_slots} slots...")

    # Pre-compute metrics
    print("Pre-computing spatial metrics...")
    precomputed_metrics = precompute_metrics(all_slot_data, mev_series, attest_series, proposal_time_series)

    # Pre-compute paper metrics
    region_metrics, country_metrics, continent_metrics = compute_paper_metrics(validator_agent_regions, region_to_country)
    precomputed_metrics["gini"] = continent_metrics["gini"].values.tolist()
    precomputed_metrics["hhi"] = continent_metrics["hhi"].values.tolist()
    precomputed_metrics["liveness"] = continent_metrics["liveness"].values.tolist()

    profits_metrics = parse_profit(profits_df)
    precomputed_metrics["profit_variance"] = profits_metrics["cv"].values.tolist()
    
    precomputed_metrics["failed_block_proposals"] = failed_block_proposals
    
    # Pre-compute info distances
    print("Pre-computing info distances...")
    if model == "SSP":
        info_data = [region_to_xyz.get(i[1], (0.0, 0.0, 0.0)) for i in relay_names]
        raw_sources = relay_names
    else:  # MSP
        info_data = [region_to_xyz.get(i[1], (0.0, 0.0, 0.0)) for i in signal_names]
        raw_sources = signal_names

    sources = normalize_sources(raw_sources, model, label_style=source_label_style)
    
    info_distances = precompute_info_distances(all_slot_data, [info_data]*len(all_slot_data))
    precomputed_metrics["info_avg_distance"] = info_distances

    # Compile all preprocessed data
    preprocessed_data = build_preprocessed_payload(
        validator_agent_regions=validator_agent_regions,
        n_slots=n_slots,
        metrics=precomputed_metrics,
        sources=sources,
        validator_count=validator_count,
        cost=cost,
        delta=delta,
        cutoff=cutoff,
        gamma=gamma,
        description=description,
    )
    
    # Save preprocessed data
    output_file = f"{output_dir}/preprocessed_data.json"
    print(f"Saving preprocessed data to {output_file}...")
    
    # Convert any numpy types to native Python types for JSON serialization
    preprocessed_data_serializable = make_json_serializable(preprocessed_data)
    
    with open(output_file, 'w') as f:
        json.dump(preprocessed_data_serializable, f)
    
    print("Data preprocessing completed successfully!")
    return preprocessed_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess simulation data for optimized loading.")
    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        default="data",
        help="Path to the data folder containing region data.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="default-simulation",
        help="Output directory where simulation results are stored.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="SSP",
        choices=["SSP", "MSP"],
        help="Simulation model type (SSP or MSP).",
    )
    parser.add_argument(
        "--validator-count",
        type=int,
        default=None,
        help="Explicit validator count for the published-style output metadata (defaults to the first slot total).",
    )
    parser.add_argument(
        "--cost",
        type=float,
        default=None,
        help="Migration cost to include in the preprocessed dataset metadata.",
    )
    parser.add_argument(
        "--delta",
        type=int,
        default=None,
        help="Slot time in milliseconds to include in the preprocessed dataset metadata.",
    )
    parser.add_argument(
        "--cutoff",
        type=int,
        default=None,
        help="Attestation cutoff time in milliseconds to include in the preprocessed dataset metadata.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Attestation threshold to include in the preprocessed dataset metadata.",
    )
    parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Optional experiment summary to include in the preprocessed dataset metadata.",
    )
    parser.add_argument(
        "--source-label-style",
        type=str,
        default="published",
        choices=["published", "raw"],
        help="How to label the sources array in the output JSON.",
    )

    
    args = parser.parse_args()
    preprocess_simulation_data(
        args.data_dir,
        args.output_dir,
        args.model,
        validator_count=args.validator_count,
        cost=args.cost,
        delta=args.delta,
        cutoff=args.cutoff,
        gamma=args.gamma,
        description=args.description,
        source_label_style=args.source_label_style,
    )
