from alpaca.data.live.option import OptionDataStream
from alpaca.data.live.stock import StockDataStream
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionChainRequest, OptionSnapshotRequest
import math, datetime, csv, json, sys, os
from alpaca_token import KEY, SECRET
sys.path.append('../')
from csv_writer import DivergenceCSVWriter
from util.calculations import (
    calculate_divergence, calculate_put_call_parity_divergence,
    stochastic_divergence_model, measure_theoretic_divergence,
    high_dimensional_divergence_analysis, advanced_change_point_detection
)
from util.realtime_divergence import DivergenceMonitor

current_spy_price = None
last_csv_update = None
last_divergence_check = None
option_client = OptionHistoricalDataClient(KEY, SECRET)
today = datetime.date.today()

#----------------------------Initialize divergence monitor and CSV writer
divergence_monitor = DivergenceMonitor(lookback_window=50, signal_threshold=2.0)
csv_writer = DivergenceCSVWriter(output_dir='../data')

#----------------------------Market data storage
market_data_buffer = []
divergence_results = []

#----------------------------Divergence alerts file
alerts_file = open(f'../data/divergence_alerts_{today.month}_{today.day}.json', 'w')

def get_option_chain_offline(spy_price):
    """Generate option symbols for analysis around current SPY price"""
    lower_bound = math.floor(spy_price * 0.99)
    upper_bound = math.ceil(spy_price * 1.01)
    calls = [f"SPY{today.strftime('%y%m%d')}C{strike:05d}000" for strike in range(lower_bound, upper_bound + 1)]
    puts = [f"SPY{today.strftime('%y%m%d')}P{strike:05d}000" for strike in range(lower_bound, upper_bound + 1)]
    return calls, puts

def process_option_data(snapshot, entry_timestamp):
    """Process option data and store for divergence analysis"""
    symbol = snapshot["symbol"]
    is_call = 'C' in symbol
    option_type = 'call' if is_call else 'put'
    strike = float(symbol.split('C' if is_call else 'P')[-1]) / 1000
    
    #------------------------Python should have null assertions ie snapshot?.bid_ask
    bid = snapshot["bid_price"]
    ask = snapshot["ask_price"]
    trade_price = (bid+ask)/2

    
    #------------------------Calculate implied volatility if we have trade price
    iv = None
    if trade_price and current_spy_price:
        #--------------------Simple IV approximation for real-time processing
        try:
            time_to_expiry = max((datetime.date(2025, 7, 9) - today).days / 365.0, 1/365)
            #----------------Simplified IV estimation
            intrinsic = max(0, current_spy_price - strike if is_call else strike - current_spy_price)
            time_value = max(0, trade_price - intrinsic)
            if time_value > 0 and time_to_expiry > 0:
                iv = time_value / (current_spy_price * math.sqrt(time_to_expiry)) * 2  # Rough approximation
        except:
            iv = None
    
    #------------------------Store market data for CSV export
    market_data_point = {
        'timestamp': entry_timestamp,
        'symbol': symbol,
        'option_type': option_type,
        'strike': strike,
        'latest_trade_price': trade_price,
        'latest_quote_bid': bid,
        'latest_quote_ask': ask,
        'implied_volatility': iv,
        'spy_price': current_spy_price
    }
    
    market_data_buffer.append(market_data_point)
    
    #------------------------Return option data for divergence analysis
    return {
        'symbol': symbol,
        'option_type': option_type,
        'strike': strike,
        'latest_trade_price': trade_price,
        'latest_quote_bid': bid,
        'latest_quote_ask': ask,
        'implied_volatility': iv
    }

def analyze_divergence(timestamp, call_data, put_data):
    """Enhanced divergence analysis with sophisticated mathematical techniques"""
    global last_divergence_check
    
    #------------------------Only check divergence every 30 seconds to avoid noise
    now = datetime.datetime.now()
    if last_divergence_check and (now - last_divergence_check).seconds < 30:
        return
    
    try:
        #--------------------Basic validation
        if not call_data or not put_data:
            return
        
        #--------------------Filter valid data - be more lenient with IV requirement for real-time processing
        valid_calls = [opt for opt in call_data if opt['latest_trade_price'] is not None]
        valid_puts = [opt for opt in put_data if opt['latest_trade_price'] is not None]
        
        if len(valid_calls) == 0 or len(valid_puts) == 0:
            return
        
        #--------------------Calculate comprehensive divergence metrics
        call_prices = [opt['latest_trade_price'] for opt in valid_calls]
        put_prices = [opt['latest_trade_price'] for opt in valid_puts]
        call_ivs = [opt['implied_volatility'] for opt in valid_calls if opt['implied_volatility']]
        put_ivs = [opt['implied_volatility'] for opt in valid_puts if opt['implied_volatility']]
        
        #--------------------Calculate multi-dimensional divergence
        divergence_metrics = None
        if len(call_prices) >= 5 and len(put_prices) >= 5:
            divergence_metrics = calculate_divergence(call_prices, put_prices)
        
        #--------------------Apply advanced mathematical techniques for larger datasets
        stochastic_analysis = None
        measure_analysis = None
        change_points = []
        
        if len(call_prices) >= 20 and len(put_prices) >= 20:
            #----------------Stochastic divergence modeling
            stochastic_analysis = stochastic_divergence_model(call_prices, put_prices)
            
            #----------------Measure-theoretic analysis
            measure_analysis = measure_theoretic_divergence(call_prices, put_prices)
            
            #----------------Change-point detection
            divergence_series = [c - p for c, p in zip(call_prices, put_prices)]
            change_points, _, _ = advanced_change_point_detection(divergence_series, method='pelt')
        
        #--------------------Add data to divergence monitor
        divergence_result = divergence_monitor.add_option_data(
            timestamp, current_spy_price, valid_calls, valid_puts
        )
        
        #--------------------Store divergence results for CSV export
        if divergence_result:
            divergence_results.append(divergence_result)
        
        if divergence_result:
            #----------------Enhance with advanced analysis results
            if stochastic_analysis and not stochastic_analysis.get('error'):
                divergence_result['stochastic_model'] = {
                    'drift_estimate': stochastic_analysis['drift_estimate'],
                    'diffusion_estimate': stochastic_analysis['diffusion_estimate'],
                    'jump_intensity': stochastic_analysis['jump_intensity'],
                    'model_type': stochastic_analysis['model_type'],
                    'jumps_detected': stochastic_analysis['jumps_detected']
                }
            
            if measure_analysis and not measure_analysis.get('error'):
                divergence_result['measure_theoretic'] = {
                    'wasserstein_distance': measure_analysis['wasserstein_distance'],
                    'jensen_shannon_divergence': measure_analysis['jensen_shannon_divergence'],
                    'optimal_transport_cost': measure_analysis['optimal_transport_cost']
                }
            
            if change_points:
                divergence_result['structural_breaks'] = {
                    'change_points_detected': len(change_points),
                    'recent_breaks': len([cp for cp in change_points if cp > len(call_prices) - 10]),
                    'regime_stability': 'unstable' if len(change_points) > 3 else 'stable'
                }
            
            #----------------Check for significant signals or extreme events
            signal = divergence_result.get('signal', 0)
            
            #----------------Enhanced signal validation using advanced techniques
            signal_confidence = 1.0
            
            #----------------Reduce confidence if model detects jumps
            if stochastic_analysis and stochastic_analysis.get('model_type') == 'jump_diffusion':
                signal_confidence *= 0.8
                
            #----------------Reduce confidence if many structural breaks detected
            if len(change_points) > 2:
                signal_confidence *= 0.7
                
            #----------------Increase confidence if measure-theoretic analysis confirms divergence
            if measure_analysis and measure_analysis.get('wasserstein_distance', 0) > 0.1:
                signal_confidence *= 1.2
            
            signal_confidence = min(signal_confidence, 1.0)
            
            if signal != 0 and signal_confidence > 0.6:
                alert = {
                    'timestamp': timestamp.isoformat(),
                    'alert_type': 'enhanced_divergence_signal',
                    'signal': signal,
                    'signal_name': {-1: 'SELL', 1: 'BUY'}.get(signal, 'NEUTRAL'),
                    'signal_confidence': signal_confidence,
                    'spy_price': current_spy_price,
                    'price_divergence': divergence_result['price_divergence'],
                    'iv_divergence': divergence_result['iv_divergence'],
                    'parity_divergence': divergence_result['parity_divergence'],
                    'mathematical_validation': {
                        'stochastic_model_type': stochastic_analysis.get('model_type') if stochastic_analysis else None,
                        'structural_breaks': len(change_points),
                        'transport_cost': measure_analysis.get('optimal_transport_cost') if measure_analysis else None
                    }
                }
                
                # Write alert to file
                alerts_file.write(json.dumps(alert) + '\n')
                alerts_file.flush()
                
                print(f"DIVERGENCE SIGNAL: {alert['signal_name']} at {timestamp}")
                print(f"  Confidence: {signal_confidence:.2f}")
                print(f"  Price Divergence: {divergence_result['price_divergence']:.4f}")
                print(f"  IV Divergence: {divergence_result.get('iv_divergence', 'N/A')}")
                
        #--------------------Periodic CSV export
        if len(market_data_buffer) >= 100:
            export_data_to_csv()
            
    except Exception as e:
        print(f"Error in divergence analysis: {e}")
    
    last_divergence_check = now

def export_data_to_csv():
    """Export buffered data to CSV files"""
    global market_data_buffer, divergence_results
    
    try:
        #--------------------Export market data
        if market_data_buffer:
            csv_writer.write_divergence_data(
                market_data_buffer, 
                filename=f'market_data_{today.month}_{today.day}_{datetime.datetime.now().strftime("%H%M")}.csv'
            )
            market_data_buffer = []  # Clear buffer
        
        #--------------------Export divergence results
        if divergence_results:
            csv_writer.write_divergence_data(
                divergence_results,
                filename=f'divergence_analysis_{today.month}_{today.day}_{datetime.datetime.now().strftime("%H%M")}.csv'
            )
            divergence_results = []  # Clear buffer
            
        print(f"Data exported to CSV at {datetime.datetime.now()}")
        
    except Exception as e:
        print(f"Error exporting data to CSV: {e}")

#----------------------------Stock data handler
async def stock_data_handler(data):
    global current_spy_price, last_csv_update
    
    for bar in data:
        """
        ('symbol', 'SPY')
        ('timestamp', datetime.datetime(2025, 7, 11, 15, 29, tzinfo=datetime.timezone.utc))
        ('open', 623.37)
        ('high', 623.37)
        ('low', 623.27)
        ('close', 623.29)
        ('volume', 695.0)
        ('trade_count', 16.0)
        ('vwap', 623.300889)
        """
        if bar[0] == "close":
            current_spy_price = bar[1]
            print(f"SPY Price Update: ${current_spy_price:.2f}")
            #----------------Export data periodically
            now = datetime.datetime.now()
            if last_csv_update is None or (now - last_csv_update).seconds > 300:  # Every 5 minutes
                export_data_to_csv()
                last_csv_update = now

#----------------------------Option data handler
async def option_data_handler(data):
    """
    ('symbol', 'SPY250711C00626000')
    ('timestamp', datetime.datetime(2025, 7, 11, 15, 52, 2, 488215, tzinfo=datetime.timezone.utc))
    ('bid_price', 0.19)
    ('bid_size', 960.0)
    ('bid_exchange', 'M')
    ('ask_price', 0.2)
    ('ask_size', 655.0)
    ('ask_exchange', 'I')
    ('conditions', ' ')
    ('tape', None)
    """
    global current_spy_price
    
    if current_spy_price is None:
        return
    
    call_data = []
    put_data = []
    timestamp = datetime.datetime.now()
    data_map = dict(data)
    option_data = process_option_data(data_map, timestamp)
    
    if option_data['option_type'] == 'call':
        call_data.append(option_data)
    else:
        put_data.append(option_data)
    
    #------------------------Analyze divergence patterns
    if call_data and put_data:
        analyze_divergence(timestamp, call_data, put_data)

#----------------------------Initialize option subscriptions after getting SPY price
async def setup_option_subscriptions(option_stream):
    while current_spy_price is None:
        await asyncio.sleep(1)
    
    calls, puts = get_option_chain_offline(current_spy_price)
    all_symbols = calls + puts
    option_stream.subscribe_quotes(option_data_handler, *all_symbols)
    print(f"Subscribed to {len(all_symbols)} option symbols")

#----------------------------Main execution
async def main():
    stock_stream = StockDataStream(KEY, SECRET)
    option_stream = OptionDataStream(KEY, SECRET)
    
    #------------------------Subscribe to SPY stock data
    stock_stream.subscribe_bars(stock_data_handler, "SPY")
    
    print("Starting real-time divergence monitoring...")
    print("Press Ctrl+C to stop")
    
    try:
        #--------------------Setup option subscriptions after stock price is available
        asyncio.create_task(setup_option_subscriptions(option_stream))
        
        #--------------------Run streams concurrently
        await asyncio.gather(
            stock_stream._run_forever(),
            option_stream._run_forever()
        )
    except KeyboardInterrupt:
        print("\nShutting down...")
        #--------------------Final data export
        export_data_to_csv()
        alerts_file.close()
        print("Data exported and files closed.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())