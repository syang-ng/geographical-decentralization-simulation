import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gmean
import copy

# ── PARAMETERS ────────────────────────────────────────────────────────────────
BUCKET_COUNT = 7
AGENT_COUNT = 50
MAX_ROUNDS = 1000
EARLY_STOP_NO_CHANGE = 1000  # stop if this many rounds in a row have no reassignments

# Pareto for initial bucket values
alpha = 2.5
x_min = 10.0

# The p-values to test
VALUE_EXPONENTS = [1.0, 0.5, 2.0]

# ── 1) Build common initial setup ─────────────────────────────────────────────
pareto_draws = (np.random.pareto(alpha, size=BUCKET_COUNT) + 1) * x_min
base_buckets = pd.DataFrame(
    {
        "id": range(BUCKET_COUNT),
        "original_value": pareto_draws,
        "subscriber_count": np.zeros(BUCKET_COUNT, dtype=int),
    }
)

# Initial weighted assignment ∝ original_value
probs = base_buckets["original_value"] / base_buckets["original_value"].sum()
base_agent_assignments = []
for agent in range(AGENT_COUNT):
    choice = np.random.choice(base_buckets["id"], p=probs)
    base_agent_assignments.append(choice)
    base_buckets.at[choice, "subscriber_count"] += 1

# ── 2) Run experiments for each exponent ───────────────────────────────────────
results = {}
for p in VALUE_EXPONENTS:
    # Deep-copy so each p starts from the same state
    buckets = base_buckets.copy(deep=True)
    agent_assignments = base_agent_assignments.copy()

    gmean_list = []
    snapshots = []
    no_change_counter = 0

    # value-computing function
    def compute_value(df, exp):
        return df["original_value"] / (df["subscriber_count"] + 1) ** exp

    # pick & reassign with change detection
    for rnd in range(1, MAX_ROUNDS + 1):
        agent = np.random.choice(AGENT_COUNT)
        old = agent_assignments[agent]
        # remove from old bucket
        buckets.at[old, "subscriber_count"] -= 1

        # recompute values
        buckets["value"] = compute_value(buckets, p)
        best = buckets["value"].idxmax()

        # if best==old, it's a no-op: revert removal
        if best == old:
            buckets.at[old, "subscriber_count"] += 1
            no_change_counter += 1
        else:
            # perform the move
            buckets.at[best, "subscriber_count"] += 1
            agent_assignments[agent] = best
            no_change_counter = 0

        # log geometric mean and snapshot
        gmean_list.append(gmean(buckets["subscriber_count"]))
        snap = buckets[["id", "subscriber_count", "value"]].copy()
        snap["round"] = rnd
        snapshots.append(snap)

        # early stopping
        if no_change_counter >= EARLY_STOP_NO_CHANGE:
            print(
                f"p={p}: no changes for {EARLY_STOP_NO_CHANGE} rounds → stopping at round {rnd}"
            )
            break

    # store results
    results[p] = {
        "gmean": np.array(gmean_list),
        "snapshots": pd.concat(snapshots, ignore_index=True),
        "rounds_run": rnd,
    }

# ── 3) Combined geometric-mean plot ────────────────────────────────────────────
plt.figure(figsize=(8, 4))
for p, data in results.items():
    rounds_done = data["rounds_run"]
    plt.plot(
        range(1, rounds_done + 1),
        data["gmean"],
        lw=1.5,
        label=f"p = {p} (ran {rounds_done} rounds)",
    )
plt.xlabel("Round")
plt.ylabel("Geometric Mean of Subscriber Counts")
plt.title("G-Mean Over Time for Different Value Exponents")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("gmean_comparison.png")
plt.close()

# ── 4) Individual charts per exponent ──────────────────────────────────────────
for p, data in results.items():
    log_df = data["snapshots"]
    rounds_done = data["rounds_run"]

    # 4a) Subscribers per bucket
    plt.figure(figsize=(10, 6))
    for bid, grp in log_df.groupby("id"):
        plt.plot(grp["round"], grp["subscriber_count"], label=f"Bucket {bid}")
    plt.xlabel("Round")
    plt.ylabel("Subscriber Count")
    plt.title(f"Subscribers Over Time (p = {p})")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"subs_p{p}.png")
    plt.close()

    # 4b) Bucket value per bucket
    plt.figure(figsize=(10, 6))
    for bid, grp in log_df.groupby("id"):
        plt.plot(grp["round"], grp["value"], label=f"Bucket {bid}")
    plt.xlabel("Round")
    plt.ylabel("Value (original_value / (subs+1)^p)")
    plt.title(f"Bucket Value Over Time (p = {p})")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"value_p{p}.png")
    plt.close()

print("Done! Generated:")
print(" • gmean_comparison.png")
for p in VALUE_EXPONENTS:
    print(f" • subs_p{p}.png, value_p{p}.png")
