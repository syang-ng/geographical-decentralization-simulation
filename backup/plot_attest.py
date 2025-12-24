from collections import defaultdict, Counter
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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

    return big_region_df, region_df


def plot_distribution(folder_path1, folder_path2):
    big_region_df1, region_df1 = load_folder(folder_path1)
    big_region_df2, region_df2 = load_folder(folder_path2)

    hue_order = ['North America', 'Europe', 'Asia', 'Middle East', 'Oceania', 'South America', 'Africa',]
    # hue_order = ['North America', 'South America', 'Europe', 'Oceania', 'Asia', 'Africa', 'Middle East']

    fig, axes = plt.subplots(1, 2, figsize=(32, 8), dpi=100, sharey=True)

    sns.set_style("whitegrid")
    ax1 = sns.lineplot(data=big_region_df1, x='slot', y='percentage', hue='region',
                    lw=8.0, hue_order=hue_order, ax=axes[0])
    ax1.set_xlabel('Slot ($c=0$)', fontsize=42)
    ax1.set_ylabel('Validator Distribution (%)', fontsize=42)
    ax1.legend_.remove()
    # ax1.set_xlim(0, 100)
    # ax1.set_ylim(0, 100)
    ax1.set_xlim(0, max(region_df1['slot'])+1)
    ax1.tick_params(labelsize=42)
    # print(big_region_df1)

    ax2 = sns.lineplot(data=big_region_df2, x='slot', y='percentage', hue='region',
                    lw=8.0, hue_order=hue_order, ax=axes[1])
    ax2.set_xlabel('Slot ($c=0.003973$)', fontsize=42)
    ax2.set_ylabel('')
    # ax2.set_ylim(0, 100)
    ax2.set_xlim(0, max(region_df2['slot'])+1)
    ax2.tick_params(labelsize=42)

    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend_.remove()
    fig.legend(handles, labels, title=None, fontsize=42, ncol=7,
            loc='upper center', bbox_to_anchor=(0.5, 1.12), 
            framealpha=0, facecolor="none", edgecolor="none", columnspacing=0.5)

    plt.tight_layout()
    plt.savefig('./marco_region_distribution_new.pdf', bbox_inches='tight')


    # import IPython; IPython.embed(colors="neutral")  # noqa: E402

    # hue_order = ['North America', 'South America', 'Europe', 'Oceania', 'Asia', 'Africa', 'Middle East', ]
    # plt.figure(figsize=(16, 8), dpi=100)
    # sns.set_style("whitegrid")
    # ax = sns.lineplot(data=big_region_df, x='slot', y='percentage', hue='region', lw=6.0, hue_order=hue_order)
    # ax.set_xlabel('Slot', fontsize=32)
    # ax.set_ylabel('Validator Distribution (%)', fontsize=32)
    # ax.legend(title=None, title_fontsize=24, fontsize=28, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.25), framealpha=0, facecolor="none", edgecolor="none")
    # plt.xlim(0, max(big_region_df['slot'])+1)
    # plt.xticks(fontsize=32)
    # plt.yticks(fontsize=32)
    # plt.tight_layout()
    # plt.savefig(folder_path + '/big_region_distribution.pdf', bbox_inches='tight')

    # freq_counter = defaultdict(int)
    # for slot, region_counts in data.items():
    #     for region, count in region_counts[:5]:
    #         freq_counter[region] += 1
    # most_frequent_regions = [region for region, _ in Counter(freq_counter).most_common(8)]
    
    # plt.figure(figsize=(16, 7), dpi=100)
    # sns.set_style("whitegrid")
    # ax = sns.lineplot(data=region_df, x='slot', y='percentage', hue='region', lw=6.0, hue_order=most_frequent_regions)
    # ax.set_xlabel('Slot', fontsize=32)
    # ax.set_ylabel('Validator Distribution (%)', fontsize=32)
    # ax.legend(title=None, title_fontsize=24, fontsize=22, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.25), framealpha=0, facecolor="none", edgecolor="none")
    # plt.xlim(0, max(region_df['slot'])+1)
    # plt.xticks(fontsize=32)
    # plt.yticks(fontsize=32)
    # plt.tight_layout()
    # plt.savefig(folder_path + '/region_distribution.pdf')


    # ax.set_title('Validator Distribution'


def plot_distribution_more(folder_path1, folder_path2, folder_path3, folder_path4):
    big_region_df1, region_df1 = load_folder(folder_path1)
    big_region_df2, region_df2 = load_folder(folder_path2)
    big_region_df3, region_df3 = load_folder(folder_path3)
    big_region_df4, region_df4 = load_folder(folder_path4)
    fig, axes = plt.subplots(2, 2, figsize=(32, 14), dpi=100, sharey=True)
    hue_order = ['North America', 'Europe', 'Asia', 'Middle East', 'Oceania', 'South America', 'Africa',]

    cost2 = folder_path2.split('_')[-1]
    cost3 = folder_path3.split('_')[-1]
    cost4 = folder_path4.split('_')[-1]

    sns.set_style("whitegrid")

    # subplot [0,0]
    ax1 = sns.lineplot(data=big_region_df1, x='slot', y='percentage', hue='region', style='region',
                    lw=8.0, hue_order=hue_order, ax=axes[0][0])
    ax1.set_xlabel('Slot (MEV-Boost, $c=0$)', fontsize=42)
    ax1.set_ylabel('Validator Distribution (%)', fontsize=42)
    ax1.legend_.remove()
    ax1.set_xlim(0, max(region_df1['slot'])+1)
    ax1.tick_params(labelsize=42)

    # subplot [0,1]
    ax2 = sns.lineplot(data=big_region_df2, x='slot', y='percentage', hue='region', style='region',
                    lw=8.0, hue_order=hue_order, ax=axes[0][1])
    ax2.set_xlabel('Slot (MEV-Boost, $c=Q_{50}$)', fontsize=42)
    # ax2.set_xlabel('Slot ($c=0.0004$)', fontsize=42)
    ax2.set_ylabel('')
    ax2.legend_.remove()
    ax2.set_xlim(0, max(region_df2['slot'])+1)
    ax2.tick_params(labelsize=42)

    ax3 = sns.lineplot(data=big_region_df3, x='slot', y='percentage', hue='region', style='region',
                    lw=8.0, hue_order=hue_order, ax=axes[1][0])
    ax3.set_xlabel('Slot (Non-MEV-Boost, $c=0$)', fontsize=42)
    # ax3.set_xlabel('Slot ($c=0.0022$)', fontsize=42)
    ax3.set_ylabel('Validator Distribution (%)', fontsize=42)
    ax3.legend_.remove()
    ax3.set_xlim(0, max(region_df3['slot'])+1)
    ax3.tick_params(labelsize=42)

    ax4 = sns.lineplot(data=big_region_df4, x='slot', y='percentage', hue='region', style='region',
                    lw=8.0, hue_order=hue_order, ax=axes[1][1])
    ax4.set_xlabel('Slot (Non-MEV-Boost, $c=Q_{50}$)', fontsize=42)
    # ax4.set_xlabel('Slot ($c=0.003973$)', fontsize=42)
    ax4.set_ylabel('')
    ax4.legend_.remove()
    ax4.set_xlim(0, max(region_df4['slot'])+1)
    ax4.tick_params(labelsize=42)

    # 统一 legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, title=None, fontsize=42, ncol=7,
            loc='upper center', bbox_to_anchor=(0.5, 1.08),
            framealpha=0, facecolor="none", edgecolor="none", columnspacing=0.5)

    plt.tight_layout()
    print(folder_path1, folder_path2, folder_path3, folder_path4)
    plt.savefig('./marco_region_distribution_new.pdf', bbox_inches='tight')

    
import sys
if __name__ == "__main__":
    folder_path1 = sys.argv[1]
    folder_path2 = sys.argv[2]
    folder_path3 = sys.argv[3]
    folder_path4 = sys.argv[4]
    # load_folder(folder_path1,)
    plot_distribution_more(folder_path1, folder_path2, folder_path3, folder_path4)