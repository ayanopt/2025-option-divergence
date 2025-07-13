from alpaca.data.live.option import OptionDataStream
from alpaca.data.live.stock import StockDataStream
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionChainRequest, OptionSnapshotRequest
import math,datetime,csv
from alpaca_token import KEY, SECRET
from util.calculations import black_scholes_greeks, get_time_to_expiry, implied_volatility

current_spy_price = None
last_csv_update = None
option_client = OptionHistoricalDataClient(KEY, SECRET)
today = datetime.date.today()
csv_file = open(f'../data/raw/option_data_{today.month}_{today.day}.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['timestamp', 'symbol', 'option_type', 'strike', 'latest_trade_price', 'latest_quote_bid', 'latest_quote_ask', 'implied_volatility', 'delta', 'gamma', 'theta', 'vega', 'rho', 'price'])
#-----------------------------------------------------------------------------
def get_option_chain_offline(spy_price):
    lower_bound = math.floor(spy_price * 0.99)
    upper_bound = math.ceil(spy_price * 1.01)
    #-------------SPY250707P00624000
    return [f"SPY{today.strftime('%y%m%d')}C{strike:05d}000" for strike in range(lower_bound, upper_bound + 1)], [f"SPY{today.strftime('%y%m%d')}P{strike:05d}000" for strike in range(lower_bound, upper_bound + 1)]
#-----------------------------------------------------------------------------
def get_option_chain(spy_price):
    lower_bound = spy_price * 0.99
    upper_bound = spy_price * 1.01
    
    request = OptionChainRequest(
        underlying_symbol="SPY",
        strike_price_gte=lower_bound,
        strike_price_lte=upper_bound,
        expiration_date=today
    )
    chain = option_client.get_option_chain(request)
    
    calls = [opt for opt in chain if 'C' in opt]
    puts = [opt for opt in chain if 'P' in opt]
    
    min_count = min(len(calls), len(puts))
    return calls[:min_count], puts[:min_count]
#-----------------------------------------------------------------------------
def flush_out_greeks(symbol, snapshot, entry_timestamp):
    is_call = 'C' in symbol
    strike = float(symbol.split('C' if is_call else 'P')[-1])

    #-------------check if otm
    #if (is_call and strike > current_spy_price) or (not is_call and strike < current_spy_price):
    #-------------   return
    option_type = 'call' if is_call else 'put'
    strike = float(symbol.split('C' if is_call else 'P')[-1]) / 1000
    
    trade_price = snapshot.latest_trade.price if snapshot.latest_trade else None
    bid = snapshot.latest_quote.bid_price if snapshot.latest_quote else None
    ask = snapshot.latest_quote.ask_price if snapshot.latest_quote else None
    
    #-------------Calculate IV and greeks using Black-Scholes
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
#-----------------------------------------------------------------------------
async def load_price(data):
    global current_spy_price, last_csv_update
    current_spy_price = data.price
    print(f"SPY Price: ${current_spy_price}")
    
    now = datetime.datetime.now()
    cadence = 10 # TODO: play around with
    if last_csv_update and (now - last_csv_update).seconds < cadence:
        return
    
    try:
        calls, puts = get_option_chain_offline(current_spy_price)
        all_symbols = calls + puts
        print("Fetching details for", calls, puts)
        
        if all_symbols:
            snapshot_request = OptionSnapshotRequest(symbol_or_symbols=all_symbols)
            snapshots = option_client.get_option_snapshot(snapshot_request)
            
            timestamp = datetime.datetime.now().isoformat()
            
            #--------Handle csv 
            for symbol, snapshot in snapshots.items():
                flush_out_greeks(symbol, snapshot, timestamp)

            csv_file.flush()
            last_csv_update = now
            print(f"Stored {len(snapshots)} option snapshots to CSV")
            
    except Exception as e:
        print(f"Error getting option data: {e}")
    
    if now.hour == 16 and now.minute >= 45:
        csv_file.close()
        await price_stream.stop_ws()

price_stream = StockDataStream(api_key=KEY,secret_key=SECRET)
price_stream.subscribe_trades(load_price,"SPY")
price_stream.run()