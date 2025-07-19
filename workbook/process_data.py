import pandas as pd
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
    spec_file = '7_15'
    filename = f'../data/raw/option_data_{spec_file}.csv'
    result_df = price_diff_x_periods(filename)
    
    #-------------Display first few rows with future price columns
    future_cols = [col for col in result_df.columns if 'price_diff' in col]
    print("Future price columns added:")
    print(result_df[['symbol', 'timestamp', 'price'] + future_cols].head(10))
    
    result_df.to_csv(f'../data/processed/option_data_with_future_prices_{spec_file}.csv', index=False)
    print(f"\nData saved to 'option_data_with_future_prices_{spec_file}.csv'")
