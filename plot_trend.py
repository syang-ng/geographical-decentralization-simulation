from collections import defaultdict, Counter
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


region_maps = {
    'australia': 'OC',
    'asia': 'AS',
    'africa': 'AF',
    'europe': 'EU',
    'northamerica': 'NA',
    'southamerica': 'SA',
    'us': 'US',
    'me': 'ME',
    "Asia": "AS",
    "Europe": "EU",
    "Africa": "AF",
    "North America": "NA",
    "South America": "SA",
    "Oceania": "OC",
    "Middle East": "ME",
    "asia": "AS",
    "europe": "EU",
    "africa": "AF",
    "north america": "NA",
    "south america": "SA",
    "oceania": "OC",
    "middle east": "ME",
}


sns.set_theme(style="whitegrid")
def load_folder(folder_path):
    with open(folder_path + '/region_counter_per_slot.json', 'r') as f:
        data = json.load(f)
    total_validators = sum([i[1] for i in data['0']])
    data = {int(k): v for k, v in data.items()}


    big_region_data = []
    region_data = []
    for key, region_counts in data.items():
        big_region_count = defaultdict(int)
        for region, count in region_counts:
            big_region = region.split('-')[0].lower()
            big_region = region_maps.get(big_region, 'Other')

            if big_region == 'Other':
                print(region, region.split('-')[0].lower())
            big_region_count[big_region] += count

            p = 100 * count / total_validators
            region_data.append({
                'slot': key,
                'region': region,
                'count': count,
                'percentage': p
            })

        for big_region, count in list(big_region_count.items()):
            p = 100 * count / total_validators
            big_region_data.append({
                'slot': key,
                'region': big_region,
                'count': count,
                'percentage': p
            })

    big_region_df = pd.DataFrame(big_region_data)

    region_df = pd.DataFrame(region_data)


    hue_order = ['NA', 'EU', 'AS', 'OC','SA', 'AF', 'ME']
    plt.figure(figsize=(16, 7), dpi=100)
    sns.set_style("whitegrid")
    ax = sns.lineplot(data=big_region_df, x='slot', y='percentage', hue='region', lw=5.0, hue_order=hue_order)
    ax.set_xlabel('Slot', fontsize=32)
    ax.set_ylabel('Validator Distribution (%)', fontsize=32)
    ax.legend(title='Region', title_fontsize=24, fontsize=20)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.tight_layout()
    plt.savefig(folder_path + '/big_region_distribution.pdf')

    freq_counter = defaultdict(int)
    for slot, region_counts in data.items():
        for region, count in region_counts[:5]:
            freq_counter[region] += 1
    most_frequent_regions = [region for region, _ in Counter(freq_counter).most_common(5)]
    
    plt.figure(figsize=(16, 7), dpi=100)
    sns.set_style("whitegrid")
    ax = sns.lineplot(data=region_df, x='slot', y='percentage', hue='region', lw=5.0, hue_order=most_frequent_regions)
    ax.set_xlabel('Slot', fontsize=32)
    ax.set_ylabel('Validator Distribution (%)', fontsize=32)
    ax.legend(title='Region', title_fontsize=24, fontsize=20)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.tight_layout()
    plt.savefig(folder_path + '/region_distribution.pdf')


    # ax.set_title('Validator Distribution'

    
import sys
if __name__ == "__main__":
    folder_path = sys.argv[1]
    load_folder(folder_path)