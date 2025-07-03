import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd

from collections import defaultdict

df = pd.read_json("output/proposer_strategy_and_mev.json")
data = []
counter = defaultdict(int)


for _, row in df.iterrows():
    strategy = row["Location_Strategy"]
    if strategy == "optimize_to_center":
        strategy = "move_to_relay"
    slot = counter[strategy]
    counter[strategy] += 1
    mev = row["MEV_Captured_Slot"]
    data.append({'strategy': strategy, 'mev': mev, 'slot': slot})

data_df = pd.DataFrame(data)

plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")
sns.scatterplot(
    data=data_df,
    x='slot',
    y='mev',
    hue='strategy',
    s=100,
    alpha=0.7
)
plt.ylim(0.45, 0.55)


plt.title("MEV Captured by Proposer Strategy Over Slots")
plt.xlabel("Slot")
plt.ylabel("MEV")
plt.tight_layout()
plt.savefig("output/mev_captured_by_proposer_strategy.png")
plt.show()