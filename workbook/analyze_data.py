import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random,math
from scipy import stats
from scipy.stats import multivariate_normal
from sklearn.covariance import EmpiricalCovariance
#-----------------------------------------------------------------------------
def load_and_group_data():
    """Load option data and group by timestamp"""
    data_dir = Path("../data/raw")
    
    #------------Load all processed data files
    all_data = []
    for file in data_dir.glob("option_data*.csv"):
        df = pd.read_csv(file)
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    #------------Convert timestamp to datetime and group
    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
    grouped = combined_df.groupby('timestamp')
    
    return grouped, combined_df
#-----------------------------------------------------------------------------
def distance_from_underlying_price(strike, underlying_price):
    """Calculate distance from underlying price (percent) between -0.01 - 0.01"""
    return ((strike - underlying_price)/underlying_price)
#-----------------------------------------------------------------------------
def apply_standardization(df):
    """
    Apply logarithmic normalization by timestamp, separately for calls and puts
    """
    df['moneyness'] = df.apply(lambda row: distance_from_underlying_price(row['strike'], row['price']), axis=1)
    df['standardized_price'] = pd.NA

    #------------Group by timestamp and apply logarithmic normalization separately for calls and puts
    normalized_dfs = []
    for timestamp, group in df.groupby('timestamp'):
        calls = group[group['option_type'] == 'call']
        puts = group[group['option_type'] == 'put']
        
        if len(calls) > 0:
            min_call = calls['latest_trade_price'].min()
            max_call = calls['latest_trade_price'].max()
            if max_call > min_call:
                calls['standardized_price'] = (np.log(calls['latest_trade_price'] + 1) - np.log(min_call + 1)) / \
                                             (np.log(max_call + 1) - np.log(min_call + 1))
        
        if len(puts) > 0:
            min_put = puts['latest_trade_price'].min()
            max_put = puts['latest_trade_price'].max()
            if max_put > min_put:
                puts['standardized_price'] = (np.log(puts['latest_trade_price'] + 1) - np.log(min_put + 1)) / \
                                            (np.log(max_put + 1) - np.log(min_put + 1))
        
        normalized_dfs.append(pd.concat([calls, puts]))
    
    if normalized_dfs:
        return pd.concat(normalized_dfs)
    return df

#-----------------------------------------------------------------------------
def plot_calls_and_puts_by_timestamp(timestamp_index=0):
    """Plot all calls on one graph and all puts on another for a specific timestamp"""
    grouped, df = load_and_group_data()
    
    #------------Get the specified timestamp
    timestamps = list(grouped.groups.keys())
    selected_timestamp = timestamps[timestamp_index]
    timestamp_data = grouped.get_group(selected_timestamp)
    
    print(f"Plotting data for timestamp: {selected_timestamp}")
    print(f"Total options at this timestamp: {len(timestamp_data)}")
    
    #------------Separate calls and puts
    calls_data = timestamp_data[timestamp_data['option_type'] == 'call']
    puts_data = timestamp_data[timestamp_data['option_type'] == 'put']
    
    print(f"Calls: {len(calls_data)}, Puts: {len(puts_data)}")
    
    #------------Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    normalized_data = apply_standardization(timestamp_data)
    
    #------------Re-separate calls and puts after standardization
    normalized_calls = normalized_data[normalized_data['option_type'] == 'call']
    normalized_puts = normalized_data[normalized_data['option_type'] == 'put']
    
    #------------Plot all calls
    if len(normalized_calls) > 0:
        calls_sorted = normalized_calls.sort_values('strike')
        ax1.plot(calls_sorted['strike'], calls_sorted['standardized_price'], 
                'bo-', linewidth=2, markersize=6, alpha=0.7)
        ax1.set_title(f'All Call Options - Normalized Price\nTimestamp: {selected_timestamp}')
        ax1.set_xlabel('Strike Price ($)')
        ax1.set_ylabel('Normalized Price (0-1)')
        ax1.grid(True, alpha=0.3)
        
        #--------Add vertical line at underlying price
        underlying_price = calls_sorted['price'].iloc[0]
        ax1.axvline(x=underlying_price, color='red', linestyle='--', linewidth=2, alpha=0.8, 
                   label=f'Underlying Price: ${underlying_price:.2f}')
        ax1.legend()
        #--------Add labels for each point
        for _, row in calls_sorted.iterrows():
            ax1.annotate(f'{row["standardized_price"]:.2f}', 
                        (row['strike'], row['standardized_price']),
                        textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    #------------Plot all puts
    if len(normalized_puts) > 0:
        puts_sorted = normalized_puts.sort_values('strike')
        ax2.plot(puts_sorted['strike'], puts_sorted['standardized_price'], 
                'ro-', linewidth=2, markersize=6, alpha=0.7)
        ax2.set_title(f'All Put Options - Normalized Price\nTimestamp: {selected_timestamp}')
        ax2.set_xlabel('Strike Price ($)')
        ax2.set_ylabel('Normalized Price (0-1)')
        ax2.grid(True, alpha=0.3)
        
        #--------Add vertical line at underlying price
        underlying_price = puts_sorted['price'].iloc[0] if len(puts_sorted) > 0 else calls_sorted['price'].iloc[0]
        ax2.axvline(x=underlying_price, color='red', linestyle='--', linewidth=2, alpha=0.8, 
                   label=f'Underlying Price: ${underlying_price:.2f}')
        ax2.legend()

        #--------Add labels for each point
        for _, row in puts_sorted.iterrows():
            ax2.annotate(f'{row["standardized_price"]:.2f}', 
                        (row['strike'], row['standardized_price']),
                        textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'calls_puts_timestamp_{timestamp_index}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nSummary Statistics:")
    print("#" + "="*60)
    if len(normalized_calls) > 0:
        print(f"\nCALLS ({len(normalized_calls)} options):")
        print(f"    Strike range: ${normalized_calls['strike'].min():.0f} - ${normalized_calls['strike'].max():.0f}")
        print(f"    Price range: ${normalized_calls['latest_trade_price'].min():.2f} - ${normalized_calls['latest_trade_price'].max():.2f}")
        print(f"    Average price: ${normalized_calls['latest_trade_price'].mean():.2f}")
        print(f"    Normalized price range: {normalized_calls['standardized_price'].min():.2f} - {normalized_calls['standardized_price'].max():.2f}")
        print(f"    Average normalized price: {normalized_calls['standardized_price'].mean():.2f}")
    
    if len(normalized_puts) > 0:
        print(f"\nPUTS ({len(normalized_puts)} options):")
        print(f"    Strike range: ${normalized_puts['strike'].min():.0f} - ${normalized_puts['strike'].max():.0f}")
        print(f"    Price range: ${normalized_puts['latest_trade_price'].min():.2f} - ${normalized_puts['latest_trade_price'].max():.2f}")
        print(f"    Average price: ${normalized_puts['latest_trade_price'].mean():.2f}")
        print(f"    Normalized price range: {normalized_puts['standardized_price'].min():.2f} - {normalized_puts['standardized_price'].max():.2f}")
        print(f"    Average normalized price: {normalized_puts['standardized_price'].mean():.2f}")

#-----------------------------------------------------------------------------
def analyze_grouped_data():
    """Analyze the grouped data structure"""
    grouped, df = load_and_group_data()
    
    print("Data Analysis:")
    print("#" + "="*60)
    print(f"Total records: {len(df)}")
    print(f"Unique timestamps: {len(grouped)}")
    print(f"Unique symbols: {len(df['symbol'].unique())}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    
    #------------Show available timestamps
    print("\nAvailable timestamps:")
    for i, ts in enumerate(list(grouped.groups.keys())[:5]):
        group_size = len(grouped.get_group(ts))
        calls = len(grouped.get_group(ts)[grouped.get_group(ts)['option_type'] == 'call'])
        puts = len(grouped.get_group(ts)[grouped.get_group(ts)['option_type'] == 'put'])
        print(f"    [{i}] {ts} - {group_size} options ({calls} calls, {puts} puts)")
    if len(grouped) > 5:
        print(f"    ... and {len(grouped) - 5} more timestamps")
    return len(grouped)

#-----------------------------------------------------------------------------
if __name__ == "__main__":
    #------------data structure
    grouped_ts = analyze_grouped_data()
    
    #------------Plot calls and puts for random timestamp
    print("\n" + "#"*70)
    ts = random.randint(0, grouped_ts - 1)
    print(f"Plotting calls and puts for {ts} timestamp...")
    plot_calls_and_puts_by_timestamp(timestamp_index=ts)
