from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca_token import KEY, SECRET
import datetime

def get_option_data(symbol):
    client = OptionHistoricalDataClient(KEY, SECRET)
    request_params = OptionBarsRequest(
                            symbol_or_symbols=[symbol],
                            start=datetime.datetime(2025,5,5),
                            start=datetime.datetime(2025,5,6),
                            timeframe=TimeFrame(1,TimeFrameUnit("Min"))
                        )
    options = client.get_option_bars(request_params)
    return options.df

print(get_option_data("SPY250117C00600000"))  # SPY call option expiring Jan 17, 2025, strike $600