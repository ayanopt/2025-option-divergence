import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random,math
#-----------------------------------------------------------------------------
def load_and_group_data():
    """Load option data and group by timestamp"""
    data_dir = Path("../data/processed")
    
    #------------Load all processed data files
    all_data = []
    for file in data_dir.glob("option_data_with_future_prices_*.csv"):
        df = pd.read_csv(file)
        all_data.append(df)
    
    #------------Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    #------------Convert timestamp to datetime and group
    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
    grouped = combined_df.groupby('timestamp')
    
    return grouped, combined_df
#-----------------------------------------------------------------------------
def delta_adjusted_price(price, delta):
    """
    Calculate delta-adjusted price for options

    The way I see it delta = dQ/dU (quote, underlying)
    and gamma = dD/dU
    """
    return price * (1/abs(delta))
#-----------------------------------------------------------------------------
def distance_from_underlying_price(strike, underlying_price):
    """Calculate distance from underlying price (percent) between -1 - 1"""
    return ((strike - underlying_price)/underlying_price)*100
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
#-----------------------------------------------------------------------------
def apply_standardization(df):
    """
    Apply standardization to the DataFrame.

    ln(quote/delta) * e^((strike - underlying_price)/strike)

    """
    df['distance_from_underlying_price'] = df.apply(lambda row: distance_from_underlying_price(row['strike'], row['price']), axis=1)
    df['delta_adjusted_price'] = df.apply(lambda row: delta_adjusted_price(row['latest_trade_price'], row['delta']), axis=1)
    df['standardized_price'] = np.where(
        df['option_type'] == 'call',
        sigmoid(np.log(df['delta_adjusted_price']) * np.exp(df['distance_from_underlying_price']+1)),
        sigmoid(np.log(df['delta_adjusted_price']) * np.exp(1 - df['distance_from_underlying_price']))
    )

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
    #------------Plot all calls
    if len(calls_data) > 0:
        calls_sorted = calls_data.sort_values('strike')
        apply_standardization(calls_sorted)
        ax1.plot(calls_sorted['strike'], calls_sorted['standardized_price'], 
                'bo-', linewidth=2, markersize=6, alpha=0.7)
        ax1.set_title(f'All Call Options - Latest Trade Price\nTimestamp: {selected_timestamp}')
        ax1.set_xlabel('Strike Price ($)')
        ax1.set_ylabel('Latest Trade Price ($)')
        ax1.grid(True, alpha=0.3)
        
        #--------Add vertical line at underlying price
        underlying_price = calls_data['price'].iloc[0]
        ax1.axvline(x=underlying_price, color='red', linestyle='--', linewidth=2, alpha=0.8, 
                   label=f'Underlying Price: ${underlying_price:.2f}')
        ax1.legend()
        #--------Add labels for each point
        for _, row in calls_sorted.iterrows():
            ax1.annotate(f'${row['standardized_price']:.2f}', 
                        (row['strike'], row['standardized_price']),
                        textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    #------------Plot all puts
    if len(puts_data) > 0:
        puts_sorted = puts_data.sort_values('strike')

        apply_standardization(puts_sorted)
        ax2.plot(puts_sorted['strike'], puts_sorted['standardized_price'], 
                'ro-', linewidth=2, markersize=6, alpha=0.7)
        ax2.set_title(f'All Put Options - Latest Trade Price\nTimestamp: {selected_timestamp}')
        ax2.set_xlabel('Strike Price ($)')
        ax2.set_ylabel('Latest Trade Price ($)')
        ax2.grid(True, alpha=0.3)
        
        #--------Add vertical line at underlying price
        underlying_price = calls_data['price'].iloc[0]
        ax2.axvline(x=underlying_price, color='red', linestyle='--', linewidth=2, alpha=0.8, 
                   label=f'Underlying Price: ${underlying_price:.2f}')
        ax2.legend()

        #--------Add labels for each point
        for _, row in puts_sorted.iterrows():
            ax2.annotate(f'${row['standardized_price']:.2f}', 
                        (row['strike'], row['standardized_price']),
                        textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'calls_puts_timestamp_{timestamp_index}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    #------------Print summary statistics
    print("\nSummary Statistics:")
    print("#" + "="*60)
    if len(calls_data) > 0:
        print(f"\nCALLS ({len(calls_data)} options):")
        print(f"    Strike range: ${calls_data['strike'].min():.0f} - ${calls_data['strike'].max():.0f}")
        print(f"    Price range: ${calls_data['latest_trade_price'].min():.2f} - ${calls_data['latest_trade_price'].max():.2f}")
        print(f"    Average price: ${calls_data['latest_trade_price'].mean():.2f}")
    
    if len(puts_data) > 0:
        print(f"\nPUTS ({len(puts_data)} options):")
        print(f"    Strike range: ${puts_data['strike'].min():.0f} - ${puts_data['strike'].max():.0f}")
        print(f"    Price range: ${puts_data['latest_trade_price'].min():.2f} - ${puts_data['latest_trade_price'].max():.2f}")
        print(f"    Average price: ${puts_data['latest_trade_price'].mean():.2f}")

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
    #------------Analyze the data structure
    grouped_ts = analyze_grouped_data()
    
    #------------Plot calls and puts for random timestamp
    print("\n" + "#"*70)
    ts = random.randint(0, grouped_ts - 1)
    print(f"Plotting calls and puts for {ts} timestamp...")
    plot_calls_and_puts_by_timestamp(timestamp_index=ts)
