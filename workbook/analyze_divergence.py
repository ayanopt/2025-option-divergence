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
    data_files = glob.glob('../data/*.csv')
    
    if not data_files:
        print("No data files found in data/ directory")
        return None
    
    all_data = []
    for file in data_files:
        df = pd.read_csv(file)
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    #------------------------Clean and process data
    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp']).dt.round('s')
    combined_df = combined_df.dropna(subset=['latest_trade_price'])
    combined_df = combined_df.sort_values(['timestamp', 'strike'])
    
    #------------------------Calculate moneyness
    combined_df['moneyness'] = np.where(
        combined_df['option_type'] == 'call',
        combined_df['strike'] / combined_df['spy_price'],
        combined_df['spy_price'] / combined_df['strike']
    )
    
    print(f"Loaded {len(combined_df)} option records")
    print(f"Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
    print(f"Strike range: {combined_df['strike'].min()} to {combined_df['strike'].max()}")
    
    return combined_df

def analyze_atm_divergence(df: pd.DataFrame):
    """Analyze divergence patterns for at-the-money options"""
    
    #------------------------Find ATM options for each timestamp
    atm_data = []
    
    for timestamp in df['timestamp'].unique():
        timestamp_data = df[df['timestamp'] == timestamp]
        
        if len(timestamp_data) == 0:
            continue
            
        spy_price = timestamp_data['spy_price'].iloc[0]
        
        #--------------------Find closest strikes to ATM for calls and puts
        calls = timestamp_data[timestamp_data['option_type'] == 'call']
        puts = timestamp_data[timestamp_data['option_type'] == 'put']

        if len(calls) == 0 or len(puts) == 0:
            print(f"No calls or puts data for timestamp {timestamp}")
            continue
        
        #--------------------Find ATM call and put
        calls['distance'] = abs(calls['strike'] - spy_price)
        puts['distance'] = abs(puts['strike'] - spy_price)
        
        atm_call = calls.loc[calls['distance'].idxmin()]
        atm_put = puts.loc[puts['distance'].idxmin()]
        #--------------------Calculate divergence metrics
        time_to_expiry = max((pd.to_datetime('2025-07-09') - pd.to_datetime(timestamp)).days / 365.0, 1/365)
        
        parity_result = calculate_put_call_parity_divergence(
            atm_call['latest_trade_price'],
            atm_put['latest_trade_price'],
            spy_price,
            atm_call['strike'],
            time_to_expiry
        )
        
        parity_divergence = parity_result['divergence'] if isinstance(parity_result, dict) else parity_result
        
        atm_data.append({
            'timestamp': timestamp,
            'spy_price': spy_price,
            'call_price': atm_call['latest_trade_price'],
            'put_price': atm_put['latest_trade_price'],
            'call_iv': atm_call.get('implied_volatility', 0),
            'put_iv': atm_put.get('implied_volatility', 0),
            'call_strike': atm_call['strike'],
            'put_strike': atm_put['strike'],
            'price_divergence': atm_call['latest_trade_price'] - atm_put['latest_trade_price'],
            'iv_divergence': (atm_call.get('implied_volatility', 0) or 0) - (atm_put.get('implied_volatility', 0) or 0),
            'parity_divergence': parity_divergence
        })
    
    return pd.DataFrame(atm_data)

def generate_divergence_signals(atm_df):
    """Generate trading signals based on divergence patterns"""
    
    if len(atm_df) < 20:
        print("Insufficient data for signal generation")
        return atm_df
    
    #------------------------Calculate rolling statistics
    window = min(20, len(atm_df) // 2)
    atm_df = atm_df.sort_values('timestamp').reset_index(drop=True)
    
    #------------------------Generate signals using adaptive method
    price_signals, price_metadata = rolling_divergence_signal(
        atm_df['price_divergence'].tolist(), 
        window=window, 
        threshold=2.0,
        method='adaptive'
    )
    
    iv_signals, iv_metadata = rolling_divergence_signal(
        atm_df['iv_divergence'].tolist(),
        window=window,
        threshold=1.5,
        method='adaptive'
    )
    
    parity_signals, parity_metadata = rolling_divergence_signal(
        atm_df['parity_divergence'].tolist(),
        window=window,
        threshold=2.0,
        method='adaptive'
    )
    
    atm_df['price_signal'] = price_signals
    atm_df['iv_signal'] = iv_signals
    atm_df['parity_signal'] = parity_signals
    
    #------------------------Combined signal with confidence weighting
    signal_weights = np.array([0.4, 0.3, 0.3])  # Price, IV, Parity
    combined_raw = (signal_weights[0] * np.array(price_signals) + 
                   signal_weights[1] * np.array(iv_signals) + 
                   signal_weights[2] * np.array(parity_signals))
    
    atm_df['combined_signal'] = np.where(np.abs(combined_raw) > 0.5, 
                                        np.sign(combined_raw), 0)
    
    #------------------------Store metadata for analysis
    atm_df['signal_confidence'] = np.abs(combined_raw)
    
    return atm_df

def create_visualizations(atm_df):
    """Create visualization plots for divergence analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('SPY Option Divergence Analysis', fontsize=16)
    
    #------------------------Plot 1: Price divergence over time
    axes[0, 0].plot(atm_df['timestamp'], atm_df['price_divergence'], 'b-', linewidth=1)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].set_title('ATM Call-Put Price Divergence')
    axes[0, 0].set_ylabel('Price Divergence ($)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    #------------------------Plot 2: IV divergence over time
    axes[0, 1].plot(atm_df['timestamp'], atm_df['iv_divergence'], 'g-', linewidth=1)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].set_title('ATM Call-Put IV Divergence')
    axes[0, 1].set_ylabel('IV Divergence')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    #------------------------Plot 3: Put-call parity divergence
    axes[1, 0].plot(atm_df['timestamp'], atm_df['parity_divergence'], 'm-', linewidth=1)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].set_title('Put-Call Parity Divergence')
    axes[1, 0].set_ylabel('Parity Divergence ($)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    #------------------------Plot 4: Trading signals
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

def calculate_statistical_significance(atm_df):
    """Calculate statistical significance of divergence patterns"""
    
    results = {}
    
    for metric in ['price_divergence', 'iv_divergence', 'parity_divergence']:
        if metric in atm_df.columns:
            values = atm_df[metric].dropna()
            if len(values) > 10:
                # Test for normality and calculate confidence intervals
                mean_val = values.mean()
                std_val = values.std()
                n = len(values)
                
                # 95% confidence interval for mean
                ci_lower = mean_val - 1.96 * std_val / np.sqrt(n)
                ci_upper = mean_val + 1.96 * std_val / np.sqrt(n)
                
                # Test if mean is significantly different from zero
                t_stat = mean_val / (std_val / np.sqrt(n)) if std_val > 0 else 0
                
                # Approximate p-value using normal distribution
                from scipy.stats import norm
                p_value = 2 * (1 - norm.cdf(abs(t_stat))) if abs(t_stat) < 10 else 0.0
                
                results[metric] = {
                    'mean': mean_val,
                    'std': std_val,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
    
    return results

def main():
    """Main analysis function"""
    print("Starting SPY Option Divergence Analysis...")
    
    #------------------------Load data
    df = load_option_data()
    if df is None:
        return
    
    #------------------------Analyze ATM divergence
    print("\nAnalyzing ATM option divergence...")
    atm_df = analyze_atm_divergence(df)
    
    if len(atm_df) == 0:
        print("No ATM data found")
        return
    
    print(f"Found {len(atm_df)} ATM option pairs")
    
    #------------------------Generate signals
    print("\nGenerating divergence signals...")
    atm_df = generate_divergence_signals(atm_df)
    
    #------------------------Print summary statistics
    print("\nDivergence Summary Statistics:")
    print(f"Price Divergence - Mean: {atm_df['price_divergence'].mean():.4f}, Std: {atm_df['price_divergence'].std():.4f}")
    print(f"IV Divergence - Mean: {atm_df['iv_divergence'].mean():.4f}, Std: {atm_df['iv_divergence'].std():.4f}")
    print(f"Parity Divergence - Mean: {atm_df['parity_divergence'].mean():.4f}, Std: {atm_df['parity_divergence'].std():.4f}")
    
    #------------------------Signal summary
    signal_counts = atm_df['combined_signal'].value_counts().sort_index()
    print(f"\nSignal Distribution:")
    for signal, count in signal_counts.items():
        signal_name = {-1: 'Sell', 0: 'Neutral', 1: 'Buy'}.get(signal, 'Unknown')
        print(f"{signal_name}: {count} ({count/len(atm_df)*100:.1f}%)")
    
    #------------------------Statistical significance analysis
    print("\nAnalyzing statistical significance...")
    significance_results = calculate_statistical_significance(atm_df)
    
    for metric, stats in significance_results.items():
        print(f"\n{metric.replace('_', ' ').title()}:")
        print(f"  Mean: {stats['mean']:.4f} (95% CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}])")
        print(f"  t-statistic: {stats['t_statistic']:.3f}, p-value: {stats['p_value']:.4f}")
        print(f"  Statistically significant: {'Yes' if stats['significant'] else 'No'}")
    
    #------------------------Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(atm_df)
    
    #------------------------Save results
    atm_df.to_csv('atm_divergence_analysis.csv', index=False)
    print("\nResults saved to 'atm_divergence_analysis.csv'")
    
    #------------------------Show extreme divergence events
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