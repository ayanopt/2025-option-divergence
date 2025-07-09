#!/usr/bin/env python3
"""
Real-time SPY Option Divergence Detection
Monitors live option data for divergence patterns and generates alerts
"""

import asyncio
import pandas as pd
import numpy as np
from collections import deque
from datetime import datetime, timedelta
import json, sys, os
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from util.calculations import (
    calculate_divergence,
    calculate_put_call_parity_divergence,
    rolling_divergence_signal
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DivergenceMonitor:
    """Real-time option divergence monitoring system"""
    
    def __init__(self, lookback_window=50, signal_threshold=2.0):
        self.lookback_window = lookback_window
        self.signal_threshold = signal_threshold
        
        # Data storage
        self.price_history = deque(maxlen=lookback_window)
        self.atm_data = deque(maxlen=lookback_window)
        self.divergence_history = deque(maxlen=lookback_window)
        
        # Signal tracking
        self.last_signal = 0
        self.signal_count = {'buy': 0, 'sell': 0, 'neutral': 0}
        self.extreme_events = []
        
        # Configuration
        self.min_data_points = 20
        self.extreme_threshold = 3.0
        
        logger.info(f"DivergenceMonitor initialized with window={lookback_window}, threshold={signal_threshold}")
    
    def add_option_data(self, timestamp, spy_price, call_data, put_data):
        """
        Add new option data point and check for divergence
        
        Args:
            timestamp: Current timestamp
            spy_price: Current SPY price
            call_data: Dictionary with call option data
            put_data: Dictionary with put option data
        """
        
        # Store price history
        self.price_history.append({
            'timestamp': timestamp,
            'spy_price': spy_price
        })
        
        # Find ATM options
        atm_call = self._find_atm_option(call_data, spy_price)
        atm_put = self._find_atm_option(put_data, spy_price)
        
        if not atm_call or not atm_put:
            logger.warning("Could not find ATM options")
            return None
        
        # Calculate divergence metrics
        divergence_data = self._calculate_divergence_metrics(
            timestamp, spy_price, atm_call, atm_put
        )
        
        # Store ATM data
        self.atm_data.append(divergence_data)
        
        # Generate signals if we have enough data
        if len(self.atm_data) >= self.min_data_points:
            signal = self._generate_signal(divergence_data)
            divergence_data['signal'] = signal
            
            # Check for extreme events
            if self._is_extreme_event(divergence_data):
                self._handle_extreme_event(divergence_data)
            
            # Log significant signals
            if signal != 0 and signal != self.last_signal:
                self._log_signal_change(divergence_data)
                self.last_signal = signal
        
        self.divergence_history.append(divergence_data)
        return divergence_data
    
    def _find_atm_option(self, option_data, spy_price):
        """Find the at-the-money option closest to current SPY price"""
        if not option_data:
            return None
        
        min_distance = float('inf')
        atm_option = None
        
        for option in option_data:
            distance = abs(option.get('strike', 0) - spy_price)
            if distance < min_distance:
                min_distance = distance
                atm_option = option
        
        return atm_option
    
    def _calculate_divergence_metrics(self, timestamp, spy_price, call_option, put_option):
        """Calculate comprehensive divergence metrics"""
        
        call_price = call_option.get('latest_trade_price', 0)
        put_price = put_option.get('latest_trade_price', 0)
        call_iv = call_option.get('implied_volatility', 0)
        put_iv = put_option.get('implied_volatility', 0)
        
        # Basic divergence
        price_divergence = call_price - put_price
        iv_divergence = call_iv - put_iv
        
        # Put-call parity divergence
        time_to_expiry = max((pd.to_datetime('2025-07-09') - pd.to_datetime(timestamp)).days / 365.0, 1/365)
        parity_divergence = calculate_put_call_parity_divergence(
            call_price, put_price, spy_price, call_option.get('strike', spy_price), time_to_expiry
        )
        
        # Delta divergence
        call_delta = call_option.get('delta', 0)
        put_delta = put_option.get('delta', 0)
        delta_divergence = abs(call_delta) - abs(put_delta)
        
        return {
            'timestamp': timestamp,
            'spy_price': spy_price,
            'call_price': call_price,
            'put_price': put_price,
            'call_iv': call_iv,
            'put_iv': put_iv,
            'call_delta': call_delta,
            'put_delta': put_delta,
            'price_divergence': price_divergence,
            'iv_divergence': iv_divergence,
            'parity_divergence': parity_divergence,
            'delta_divergence': delta_divergence,
            'call_strike': call_option.get('strike', 0),
            'put_strike': put_option.get('strike', 0)
        }
    
    def _generate_signal(self, current_data):
        """Generate trading signal based on divergence patterns"""
        
        if len(self.atm_data) < self.min_data_points:
            return 0
        
        # Extract recent divergence values
        recent_price_div = [d['price_divergence'] for d in list(self.atm_data)[-self.min_data_points:]]
        recent_iv_div = [d['iv_divergence'] for d in list(self.atm_data)[-self.min_data_points:]]
        recent_parity_div = [d['parity_divergence'] for d in list(self.atm_data)[-self.min_data_points:]]
        
        # Generate individual signals
        price_signals = rolling_divergence_signal(recent_price_div, window=min(10, len(recent_price_div)), threshold=self.signal_threshold)
        iv_signals = rolling_divergence_signal(recent_iv_div, window=min(10, len(recent_iv_div)), threshold=1.5)
        parity_signals = rolling_divergence_signal(recent_parity_div, window=min(10, len(recent_parity_div)), threshold=self.signal_threshold)
        
        # Combined signal (majority vote)
        if len(price_signals) > 0 and len(iv_signals) > 0 and len(parity_signals) > 0:
            combined_signal = np.sign(price_signals[-1] + iv_signals[-1] + parity_signals[-1])
        else:
            combined_signal = 0
        
        # Update signal counts
        signal_name = {-1: 'sell', 0: 'neutral', 1: 'buy'}.get(combined_signal, 'neutral')
        self.signal_count[signal_name] += 1
        
        return int(combined_signal)
    
    def _is_extreme_event(self, data):
        """Check if current data represents an extreme divergence event"""
        
        if len(self.atm_data) < self.min_data_points:
            return False
        
        # Calculate z-scores for recent data
        recent_data = list(self.atm_data)[-self.min_data_points:]
        
        price_divs = [d['price_divergence'] for d in recent_data]
        iv_divs = [d['iv_divergence'] for d in recent_data]
        
        price_mean = np.mean(price_divs)
        price_std = np.std(price_divs)
        iv_mean = np.mean(iv_divs)
        iv_std = np.std(iv_divs)
        
        if price_std == 0 or iv_std == 0:
            return False
        
        price_zscore = abs((data['price_divergence'] - price_mean) / price_std)
        iv_zscore = abs((data['iv_divergence'] - iv_mean) / iv_std)
        
        return price_zscore > self.extreme_threshold or iv_zscore > self.extreme_threshold
    
    def _handle_extreme_event(self, data):
        """Handle extreme divergence events"""
        
        event = {
            'timestamp': data['timestamp'],
            'type': 'extreme_divergence',
            'price_divergence': data['price_divergence'],
            'iv_divergence': data['iv_divergence'],
            'parity_divergence': data['parity_divergence'],
            'spy_price': data['spy_price']
        }
        
        self.extreme_events.append(event)
        
        logger.warning(f"EXTREME DIVERGENCE EVENT: {event}")
        
        # Keep only recent extreme events
        if len(self.extreme_events) > 100:
            self.extreme_events = self.extreme_events[-50:]
    
    def _log_signal_change(self, data):
        """Log significant signal changes"""
        
        signal_name = {-1: 'SELL', 0: 'NEUTRAL', 1: 'BUY'}.get(data.get('signal', 0), 'UNKNOWN')
        
        logger.info(f"SIGNAL CHANGE: {signal_name} | "
                   f"Price Div: {data['price_divergence']:.4f} | "
                   f"IV Div: {data['iv_divergence']:.4f} | "
                   f"Parity Div: {data['parity_divergence']:.4f} | "
                   f"SPY: ${data['spy_price']:.2f}")
    
    def get_current_status(self):
        """Get current monitoring status"""
        
        if len(self.atm_data) == 0:
            return {'status': 'no_data'}
        
        latest = self.atm_data[-1]
        
        return {
            'status': 'active',
            'data_points': len(self.atm_data),
            'latest_timestamp': latest['timestamp'],
            'current_signal': latest.get('signal', 0),
            'signal_counts': self.signal_count.copy(),
            'extreme_events_count': len(self.extreme_events),
            'latest_divergence': {
                'price': latest['price_divergence'],
                'iv': latest['iv_divergence'],
                'parity': latest['parity_divergence']
            }
        }
    
    def export_data(self, filename=None):
        """Export collected data to CSV"""
        
        if not filename:
            filename = f"../../data/divergence_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        if len(self.atm_data) == 0:
            logger.warning("No data to export")
            return None
        
        df = pd.DataFrame(list(self.atm_data))
        df.to_csv(filename, index=False)
        
        logger.info(f"Exported {len(df)} records to {filename}")
        return filename

# Example usage and testing
async def simulate_realtime_monitoring():
    """Simulate real-time monitoring with sample data"""
    
    monitor = DivergenceMonitor(lookback_window=30, signal_threshold=2.0)
    
    # Simulate some option data
    base_spy_price = 623.0
    
    for i in range(100):
        # Simulate price movement
        spy_price = base_spy_price + np.random.normal(0, 0.5)
        timestamp = datetime.now() + timedelta(seconds=i*10)
        
        # Simulate call and put data
        call_data = [{
            'strike': spy_price + np.random.uniform(-2, 2),
            'latest_trade_price': max(0.1, np.random.uniform(0.5, 3.0)),
            'implied_volatility': np.random.uniform(0.15, 0.35),
            'delta': np.random.uniform(0.3, 0.7)
        }]
        
        put_data = [{
            'strike': spy_price + np.random.uniform(-2, 2),
            'latest_trade_price': max(0.1, np.random.uniform(0.5, 3.0)),
            'implied_volatility': np.random.uniform(0.15, 0.35),
            'delta': -np.random.uniform(0.3, 0.7)
        }]
        
        # Add data to monitor
        result = monitor.add_option_data(timestamp, spy_price, call_data, put_data)
        
        if result and i % 10 == 0:
            status = monitor.get_current_status()
            print(f"Status update: {status}")
        
        # Small delay to simulate real-time
        await asyncio.sleep(0.1)
    
    # Export final data
    filename = monitor.export_data()
    print(f"Simulation complete. Data exported to {filename}")
    
    return monitor

if __name__ == "__main__":
    # Run simulation
    asyncio.run(simulate_realtime_monitoring())