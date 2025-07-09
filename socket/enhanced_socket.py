from alpaca.data.live.option import OptionDataStream
from alpaca.data.live.stock import StockDataStream
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionChainRequest, OptionSnapshotRequest
import math, datetime, csv, json
from alpaca_token import KEY, SECRET
from util.calculations import black_scholes_greeks, get_time_to_expiry, implied_volatility
from util.realtime_divergence import DivergenceMonitor

current_spy_price = None
last_csv_update = None
last_divergence_check = None
option_client = OptionHistoricalDataClient(KEY, SECRET)
today = datetime.date.today()

# Initialize divergence monitor
divergence_monitor = DivergenceMonitor(lookback_window=50, signal_threshold=2.0)

# CSV setup
csv_file = open(f'../data/option_data_{today.month}_{today.day}.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['timestamp', 'symbol', 'option_type', 'strike', 'latest_trade_price', 'latest_quote_bid', 'latest_quote_ask', 'implied_volatility', 'delta', 'gamma', 'theta', 'vega', 'rho', 'price'])

# Divergence alerts file
alerts_file = open(f'../data/divergence_alerts_{today.month}_{today.day}.json', 'w')

def get_option_chain_offline(spy_price):
    lower_bound = math.floor(spy_price * 0.99)
    upper_bound = math.ceil(spy_price * 1.01)
    return [f"SPY{today.strftime('%y%m%d')}C{strike:05d}000" for strike in range(lower_bound, upper_bound + 1)], [f"SPY{today.strftime('%y%m%d')}P{strike:05d}000" for strike in range(lower_bound, upper_bound + 1)]

def flush_out_greeks(symbol, snapshot, entry_timestamp):
    is_call = 'C' in symbol
    strike = float(symbol.split('C' if is_call else 'P')[-1])

    option_type = 'call' if is_call else 'put'
    strike = float(symbol.split('C' if is_call else 'P')[-1]) / 1000
    
    trade_price = snapshot.latest_trade.price if snapshot.latest_trade else None
    bid = snapshot.latest_quote.bid_price if snapshot.latest_quote else None
    ask = snapshot.latest_quote.ask_price if snapshot.latest_quote else None
    
    # Calculate IV and greeks using Black-Scholes
    expiry_str = symbol[3:9]  # Extract YYMMDD from symbol
    T = get_time_to_expiry(f"20{expiry_str}")
    r = 0.05  # Risk-free rate
    
    if trade_price:
        iv = implied_volatility(current_spy_price, strike, T, r, trade_price, option_type)
        if iv:
            greeks = black_scholes_greeks(current_spy_price, strike, T, r, iv, option_type)
            delta = greeks['delta']
            gamma = greeks['gamma']
            theta = greeks['theta']
            vega = greeks['vega']
            rho = greeks['rho']
        else:
            delta = gamma = theta = vega = rho = None
    else:
        iv = delta = gamma = theta = vega = rho = None
    
    csv_writer.writerow([entry_timestamp, symbol, option_type, strike, trade_price, bid, ask, iv, delta, gamma, theta, vega, rho, current_spy_price])
    
    # Return option data for divergence analysis
    return {
        'symbol': symbol,
        'option_type': option_type,
        'strike': strike,
        'latest_trade_price': trade_price,
        'latest_quote_bid': bid,
        'latest_quote_ask': ask,
        'implied_volatility': iv,
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }

def analyze_divergence(timestamp, call_data, put_data):
    """Analyze option divergence and generate alerts"""
    global last_divergence_check
    
    # Only check divergence every 30 seconds to avoid noise
    now = datetime.datetime.now()
    if last_divergence_check and (now - last_divergence_check).seconds < 30:
        return
    
    try:
        # Filter valid data
        valid_calls = [opt for opt in call_data if opt['latest_trade_price'] is not None and opt['implied_volatility'] is not None]
        valid_puts = [opt for opt in put_data if opt['latest_trade_price'] is not None and opt['implied_volatility'] is not None]
        
        if len(valid_calls) == 0 or len(valid_puts) == 0:
            return
        
        # Add data to divergence monitor
        divergence_result = divergence_monitor.add_option_data(
            timestamp, current_spy_price, valid_calls, valid_puts
        )
        
        if divergence_result:
            # Check for significant signals or extreme events
            signal = divergence_result.get('signal', 0)
            
            if signal != 0:
                alert = {
                    'timestamp': timestamp.isoformat(),
                    'alert_type': 'divergence_signal',
                    'signal': signal,
                    'signal_name': {-1: 'SELL', 1: 'BUY'}.get(signal, 'NEUTRAL'),
                    'spy_price': current_spy_price,
                    'price_divergence': divergence_result['price_divergence'],
                    'iv_divergence': divergence_result['iv_divergence'],
                    'parity_divergence': divergence_result['parity_divergence']
                }
                
                # Write alert to file
                alerts_file.write(json.dumps(alert) + '\n')
                alerts_file.flush()
                
                print(f"DIVERGENCE ALERT: {alert['signal_name']} | "
                      f"Price Div: {alert['price_divergence']:.4f} | "
                      f"IV Div: {alert['iv_divergence']:.4f} | "
                      f"SPY: ${current_spy_price:.2f}")
            
            # Log status every few minutes
            if now.minute % 5 == 0 and now.second < 30:
                status = divergence_monitor.get_current_status()
                print(f"Monitor Status: {status['data_points']} points | "
                      f"Signals: Buy={status['signal_counts']['buy']}, "
                      f"Sell={status['signal_counts']['sell']}, "
                      f"Neutral={status['signal_counts']['neutral']} | "
                      f"Extreme Events: {status['extreme_events_count']}")
        
        last_divergence_check = now
        
    except Exception as e:
        print(f"Error in divergence analysis: {e}")

async def load_price(data):
    global current_spy_price, last_csv_update
    current_spy_price = data.price
    print(f"SPY Price: ${current_spy_price}")
    
    now = datetime.datetime.now()
    cadence = 10  # seconds between option data fetches
    
    if last_csv_update and (now - last_csv_update).seconds < cadence:
        return
    
    try:
        calls, puts = get_option_chain_offline(current_spy_price)
        all_symbols = calls + puts
        print("Fetching details for", calls, puts)
        
        if all_symbols:
            snapshot_request = OptionSnapshotRequest(symbol_or_symbols=all_symbols)
            snapshots = option_client.get_option_snapshot(snapshot_request)
            
            timestamp = datetime.datetime.now()
            
            # Process options and collect data for divergence analysis
            call_data = []
            put_data = []
            
            for symbol, snapshot in snapshots.items():
                option_data = flush_out_greeks(symbol, snapshot, timestamp)
                
                if option_data['option_type'] == 'call':
                    call_data.append(option_data)
                else:
                    put_data.append(option_data)

            csv_file.flush()
            last_csv_update = now
            print(f"Stored {len(snapshots)} option snapshots to CSV")
            
            # Analyze divergence patterns
            analyze_divergence(timestamp, call_data, put_data)
            
    except Exception as e:
        print(f"Error getting option data: {e}")
    
    # Check if market is closing
    if now.hour == 16 and now.minute >= 45:
        # Export divergence data before closing
        export_file = divergence_monitor.export_data()
        print(f"Exported divergence data to {export_file}")
        
        # Close files
        csv_file.close()
        alerts_file.close()
        
        # Stop the stream
        await price_stream.stop_ws()

# Enhanced monitoring with additional features
def print_startup_info():
    print("SPY Option Divergence Monitor Starting...")
    print(f"Date: {today}")
    print(f"Divergence Monitor: Window={divergence_monitor.lookback_window}, Threshold={divergence_monitor.signal_threshold}")
    print(f"Data files: option_data_{today.month}_{today.day}.csv, divergence_alerts_{today.month}_{today.day}.json")
    print("Monitoring for divergence patterns...")
    print("-" * 60)

# Start the enhanced monitoring system
print_startup_info()

price_stream = StockDataStream(api_key=KEY, secret_key=SECRET)
price_stream.subscribe_trades(load_price, "SPY")
price_stream.run()