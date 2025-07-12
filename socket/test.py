import math
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca_token import KEY, SECRET
import datetime
today = datetime.date.today()
def get_option_chain_offline(spy_price):
    lower_bound = math.floor(spy_price * 0.99)
    upper_bound = math.ceil(spy_price * 1.01)
    #------------------------SPY250707P00624000
    return [f"SPY{today.strftime('%y%m%d')}C{strike:05d}000" for strike in range(lower_bound, upper_bound + 1)], [f"SPY{today.strftime('%y%m%d')}P{strike:05d}000" for strike in range(lower_bound, upper_bound + 1)]


def get_option_data(symbol):
    client = OptionHistoricalDataClient(KEY, SECRET)
    request_params = OptionBarsRequest(
                            symbol_or_symbols=[symbol],
                            start=datetime.datetime(2025,5,6),
                            timeframe=TimeFrame(1,TimeFrameUnit("Min"))
                        )
    options = client.get_option_bars(request_params)
    return options.df

#print(get_option_data("SPY250117C00600000"))  # SPY call option expiring Jan 17, 2025, strike $600

print(get_option_chain_offline(624))