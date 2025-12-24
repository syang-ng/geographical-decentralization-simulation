import psycopg
import pandas as pd

POSTGRES_DB_URL = "postgresql://readonlyuser:DnVk10S$bRQBscPo!4epUv9mBXLvmX@merkleroot.cs.yale.internal:5432/mevboostdb"



def get_proposer_fee_recipient(begin_slot, end_slot):
    sql = f"""select slot, relay, value, proposer_fee_recipient from payloads where slot between {begin_slot} and {end_slot};"""
    
    with psycopg.connect(POSTGRES_DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            results = []
            for row in rows:
                results.append({
                    "slot": row[0],
                    "relay": row[1],
                    "value": row[2],
                    "proposer_fee_recipient": row[3],
                })
    return results


begin_slot = 10738799
end_slot = 12041998

data = get_proposer_fee_recipient(begin_slot, end_slot)
df = pd.DataFrame(data)

relays = ["Flashbots", "ultra sound relay", "BloXroute Max Profit"]
df["value"] = df["value"].astype(float) / 1e18
sub_df = df[df['relay'].isin(relays)].reindex()
for relay, relay_df in sub_df.groupby('relay'):
    total_value = relay_df['value'].sum()
    print(f"Relay: {relay}, Total Value: {total_value}")

for relay, relay_df in sub_df.groupby('relay'):
    avg_value = relay_df['value'].mean()
    median_value = relay_df['value'].median()
    print(f"Relay: {relay}, Average Value per Slot: {avg_value}, Median Value per Slot: {median_value}")


import seaborn as sns
import matplotlib.pyplot as plt


plt.figure(figsize=(12, 6))
sns.ecdfplot(data=sub_df, x='value', hue='relay', stat='proportion')
plt.xscale('log')
plt.xlim(1e-02, 1)
plt.xlabel('Bid Value (ETH)')
plt.ylabel('Cumulative Distribution Function (CDF)')
plt.savefig('relay_bid_value_cdf.png')

