#!/usr/bin/env python3
"""
SPY Option Divergence Analysis Script
Analyzes collected option data for divergence patterns and generates signals
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import glob
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../socket'))
from util.calculations import (
    calculate_divergence, 
    calculate_put_call_parity_divergence,
    rolling_divergence_signal,
    pearson_correlation
)

def load_option_data():
    """Load and combine all option data files"""
    data_files = glob.glob('../data/option_data_7_8_1.csv')
    
    if not data_files:
        print("No data files found in data/ directory")
        return None
    
    all_data = []
    for file in data_files:
        df = pd.read_csv(file)
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Clean and process data
    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
    combined_df = combined_df.dropna(subset=['latest_trade_price', 'implied_volatility'])
    combined_df = combined_df.sort_values(['timestamp', 'strike'])
    
    # Calculate moneyness
    combined_df['moneyness'] = np.where(
        combined_df['option_type'] == 'call',
        combined_df['strike'] / combined_df['price'],
        combined_df['price'] / combined_df['strike']
    )
    
    print(f"Loaded {len(combined_df)} option records")
    print(f"Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
    print(f"Strike range: {combined_df['strike'].min()} to {combined_df['strike'].max()}")
    
    return combined_df

def analyze_atm_divergence(df):
    """Analyze divergence patterns for at-the-money options"""
    
    # Find ATM options for each timestamp
    atm_data = []
    
    for timestamp in df['timestamp'].unique():
        timestamp_data = df[df['timestamp'] == timestamp]
        
        if len(timestamp_data) == 0:
            continue
            
        spy_price = timestamp_data['price'].iloc[0]
        
        # Find closest strikes to ATM for calls and puts
        calls = timestamp_data[timestamp_data['option_type'] == 'call']
        puts = timestamp_data[timestamp_data['option_type'] == 'put']
        
        if len(calls) == 0 or len(puts) == 0:
            continue
        
        # Find ATM call and put
        calls['distance'] = abs(calls['strike'] - spy_price)
        puts['distance'] = abs(puts['strike'] - spy_price)
        
        atm_call = calls.loc[calls['distance'].idxmin()]
        atm_put = puts.loc[puts['distance'].idxmin()]
        
        # Calculate divergence metrics
        time_to_expiry = max((pd.to_datetime('2025-07-09') - pd.to_datetime(timestamp)).days / 365.0, 1/365)
        
        parity_divergence = calculate_put_call_parity_divergence(
            atm_call['latest_trade_price'],
            atm_put['latest_trade_price'],
            spy_price,
            atm_call['strike'],
            time_to_expiry
        )
        
        atm_data.append({
            'timestamp': timestamp,
            'spy_price': spy_price,
            'call_price': atm_call['latest_trade_price'],
            'put_price': atm_put['latest_trade_price'],
            'call_iv': atm_call['implied_volatility'],
            'put_iv': atm_put['implied_volatility'],
            'call_delta': atm_call['delta'],
            'put_delta': atm_put['delta'],
            'call_strike': atm_call['strike'],
            'put_strike': atm_put['strike'],
            'price_divergence': atm_call['latest_trade_price'] - atm_put['latest_trade_price'],
            'iv_divergence': atm_call['implied_volatility'] - atm_put['implied_volatility'],
            'parity_divergence': parity_divergence
        })
    
    return pd.DataFrame(atm_data)

def generate_divergence_signals(atm_df):
    """Generate trading signals based on divergence patterns"""
    
    if len(atm_df) < 20:
        print("Insufficient data for signal generation")
        return atm_df
    
    # Calculate rolling statistics
    window = 20
    atm_df = atm_df.sort_values('timestamp').reset_index(drop=True)
    
    # Price divergence signals
    price_signals = rolling_divergence_signal(
        atm_df['price_divergence'].tolist(), 
        window=window, 
        threshold=2.0
    )
    
    # IV divergence signals  
    iv_signals = rolling_divergence_signal(
        atm_df['iv_divergence'].tolist(),
        window=window,
        threshold=1.5
    )
    
    # Put-call parity signals
    parity_signals = rolling_divergence_signal(
        atm_df['parity_divergence'].tolist(),
        window=window,
        threshold=2.0
    )
    
    atm_df['price_signal'] = price_signals
    atm_df['iv_signal'] = iv_signals
    atm_df['parity_signal'] = parity_signals
    
    # Combined signal (majority vote)
    atm_df['combined_signal'] = np.sign(
        atm_df['price_signal'] + atm_df['iv_signal'] + atm_df['parity_signal']
    )
    
    return atm_df

def create_visualizations(atm_df):
    """Create visualization plots for divergence analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('SPY Option Divergence Analysis', fontsize=16)
    
    # Plot 1: Price divergence over time
    axes[0, 0].plot(atm_df['timestamp'], atm_df['price_divergence'], 'b-', linewidth=1)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].set_title('ATM Call-Put Price Divergence')
    axes[0, 0].set_ylabel('Price Divergence ($)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: IV divergence over time
    axes[0, 1].plot(atm_df['timestamp'], atm_df['iv_divergence'], 'g-', linewidth=1)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].set_title('ATM Call-Put IV Divergence')
    axes[0, 1].set_ylabel('IV Divergence')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Put-call parity divergence
    axes[1, 0].plot(atm_df['timestamp'], atm_df['parity_divergence'], 'm-', linewidth=1)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].set_title('Put-Call Parity Divergence')
    axes[1, 0].set_ylabel('Parity Divergence ($)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 4: Trading signals
    signal_colors = {-1: 'red', 0: 'gray', 1: 'green'}
    for signal_val in [-1, 0, 1]:
        mask = atm_df['combined_signal'] == signal_val
        if mask.any():
            axes[1, 1].scatter(
                atm_df.loc[mask, 'timestamp'], 
                atm_df.loc[mask, 'spy_price'],
                c=signal_colors[signal_val], 
                alpha=0.7,
                label=f'Signal {signal_val}'
            )
    
    axes[1, 1].plot(atm_df['timestamp'], atm_df['spy_price'], 'k-', alpha=0.3, linewidth=0.5)
    axes[1, 1].set_title('Trading Signals on SPY Price')
    axes[1, 1].set_ylabel('SPY Price ($)')
    axes[1, 1].legend()
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('divergence_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_correlation_patterns(df):
    """Analyze correlation patterns between different metrics"""
    
    # Group by time periods to analyze correlation evolution
    df['hour'] = df['timestamp'].dt.hour
    df['minute_bucket'] = (df['timestamp'].dt.minute // 10) * 10
    
    correlation_data = []
    
    for hour in df['hour'].unique():
        hour_data = df[df['hour'] == hour]
        
        if len(hour_data) < 10:
            continue
            
        calls = hour_data[hour_data['option_type'] == 'call']
        puts = hour_data[hour_data['option_type'] == 'put']
        
        if len(calls) < 5 or len(puts) < 5:
            continue
        
        # Calculate correlations
        price_corr = pearson_correlation(
            calls['latest_trade_price'].tolist(),
            puts['latest_trade_price'].tolist()
        ) if len(calls) == len(puts) else np.nan
        
        iv_corr = pearson_correlation(
            calls['implied_volatility'].tolist(),
            puts['implied_volatility'].tolist()
        ) if len(calls) == len(puts) else np.nan
        
        correlation_data.append({
            'hour': hour,
            'price_correlation': price_corr,
            'iv_correlation': iv_corr,
            'call_count': len(calls),
            'put_count': len(puts)
        })
    
    corr_df = pd.DataFrame(correlation_data)
    
    if len(corr_df) > 0:
        print("\nCorrelation Analysis by Hour:")
        print(corr_df.to_string(index=False))
    
    return corr_df

def main():
    """Main analysis function"""
    print("Starting SPY Option Divergence Analysis...")
    
    # Load data
    df = load_option_data()
    if df is None:
        return
    
    # Analyze ATM divergence
    print("\nAnalyzing ATM option divergence...")
    atm_df = analyze_atm_divergence(df)
    
    if len(atm_df) == 0:
        print("No ATM data found")
        return
    
    print(f"Found {len(atm_df)} ATM option pairs")
    
    # Generate signals
    print("\nGenerating divergence signals...")
    atm_df = generate_divergence_signals(atm_df)
    
    # Print summary statistics
    print("\nDivergence Summary Statistics:")
    print(f"Price Divergence - Mean: {atm_df['price_divergence'].mean():.4f}, Std: {atm_df['price_divergence'].std():.4f}")
    print(f"IV Divergence - Mean: {atm_df['iv_divergence'].mean():.4f}, Std: {atm_df['iv_divergence'].std():.4f}")
    print(f"Parity Divergence - Mean: {atm_df['parity_divergence'].mean():.4f}, Std: {atm_df['parity_divergence'].std():.4f}")
    
    # Signal summary
    signal_counts = atm_df['combined_signal'].value_counts().sort_index()
    print(f"\nSignal Distribution:")
    for signal, count in signal_counts.items():
        signal_name = {-1: 'Sell', 0: 'Neutral', 1: 'Buy'}.get(signal, 'Unknown')
        print(f"{signal_name}: {count} ({count/len(atm_df)*100:.1f}%)")
    
    # Analyze correlations
    print("\nAnalyzing correlation patterns...")
    corr_df = analyze_correlation_patterns(df)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(atm_df)
    
    # Save results
    atm_df.to_csv('atm_divergence_analysis.csv', index=False)
    print("\nResults saved to 'atm_divergence_analysis.csv'")
    
    # Show extreme divergence events
    extreme_events = atm_df[
        (abs(atm_df['price_divergence']) > atm_df['price_divergence'].std() * 2) |
        (abs(atm_df['iv_divergence']) > atm_df['iv_divergence'].std() * 2)
    ]
    
    if len(extreme_events) > 0:
        print(f"\nExtreme Divergence Events ({len(extreme_events)} found):")
        print(extreme_events[['timestamp', 'price_divergence', 'iv_divergence', 'parity_divergence', 'combined_signal']].to_string(index=False))
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()