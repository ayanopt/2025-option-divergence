import pandas as pd
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
    standardized_dfs = []
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
        
        standardized_dfs.append(pd.concat([calls, puts]))
    
    if standardized_dfs:
        return pd.concat(standardized_dfs)
    return df

#-----------------------------------------------------------------------------
def price_diff_x_periods(csv_file_path, periods=[3,6,12,15,30]):
    """
    Calculate average price for future periods for each option.
    
    Args:
        csv_file_path: Path to the CSV file
        periods: List of periods to look forward (default: [6, 15, 30, 60, 180])
    
    Returns:
        DataFrame with original data plus future price columns
    """
    df = pd.read_csv(csv_file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = apply_standardization(df)
    for period in periods:
        df[f'price_diff_{period}_periods'] = None
    
    #-----------Group by symbol to calculate future prices for each option
    for symbol in df['symbol'].unique():
        symbol_mask = df['symbol'] == symbol
        symbol_data = df[symbol_mask].copy()
        
        for i in range(len(symbol_data)):
            for period in periods:
                # Get the next 'period' number of rows
                end_idx = min(i + period + 1, len(symbol_data))
                if end_idx > i + 1:  # Ensure we have future data
                    future_prices = symbol_data.iloc[i+1:end_idx]['latest_trade_price']
                    if len(future_prices) > 0:
                        avg_price = future_prices.mean()
                        # Get percent diff
                        df.loc[symbol_data.index[i], f'price_diff_{period}_periods'] = avg_price/symbol_data.iloc[i]['latest_trade_price']
    
    return df
#-----------------------------------------------------------------------------
if __name__ == "__main__":
    spec_file = '7_9'
    filename = f'../data/raw/option_data_{spec_file}.csv'
    result_df = price_diff_x_periods(filename)
    
    #-------------Display first few rows with future price columns
    future_cols = [col for col in result_df.columns if 'price_diff' in col]
    print("Future price columns added:")
    print(result_df[['symbol', 'timestamp', 'price'] + future_cols].head(10))
    
    result_df.to_csv(f'../data/processed/option_data_with_future_prices_{spec_file}.csv', index=False)
    print(f"\nData saved to 'option_data_with_future_prices_{spec_file}.csv'")
