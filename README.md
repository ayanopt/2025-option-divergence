# SPY Option Divergence Analysis System

A comprehensive real-time system for detecting and analyzing divergence patterns in SPY options, built with Python and R.

## Features

- **Real-time Data Collection**: Stream live SPY prices and option data using Alpaca API
- **Divergence Detection**: Advanced algorithms to identify call-put divergence patterns
- **Signal Generation**: Automated trading signals based on divergence metrics
- **Live Dashboard**: Real-time visualization of divergence patterns and alerts
- **Historical Analysis**: R-based statistical analysis of collected data
- **Alert System**: JSON-based alert logging for extreme divergence events

## Project Structure

```
2025-option-divergence/
├── data/                          # Data storage directory
│   ├── option_data_*.csv         # Historical option data
│   └── divergence_alerts_*.json  # Real-time alerts
├── socket/                        # Real-time data collection
│   ├── run_socket.py             # Basic option data streaming
│   ├── enhanced_socket.py         # Enhanced streaming with divergence
│   ├── util/
│   │   └── calculations.py        # Black-Scholes and divergence calculations
│   └── alpaca_token.py           # API credentials (create this)
├── workbook/
│   └── exploration.rmd           # R analysis notebook
├── analyze_divergence.py          # Historical data analysis
├── realtime_divergence.py         # Real-time divergence monitoring
├── dashboard.py                   # Streamlit dashboard
└── requirements.txt               # Python dependencies
```

## Setup

### 1. Install Dependencies

```bash
# Python dependencies
pip install -r requirements.txt

# R dependencies (for analysis notebook)
# Install R, then in R console:
install.packages(c("tidyverse", "lubridate", "plotly", "corrplot", "DT", "zoo", "gridExtra"))
```

### 2. Configure API Access

Create `socket/alpaca_token.py`:
```python
KEY = "your_alpaca_api_key"
SECRET = "your_alpaca_secret_key"
```

### 3. Create Data Directory

```bash
mkdir -p data
```

## Usage

### Real-time Data Collection

#### Basic Option Data Streaming
```bash
cd socket
python run_socket.py
```

#### Enhanced Streaming with Divergence Detection
```bash
cd socket
python enhanced_socket.py
```

This will:
- Stream live SPY prices
- Collect option data every 10 seconds
- Calculate Greeks and implied volatility
- Detect divergence patterns in real-time
- Generate alerts for extreme events
- Save data to CSV and JSON files

### Historical Analysis

#### Python Analysis
```bash
python analyze_divergence.py
```

This script will:
- Load all collected option data
- Analyze ATM option divergence patterns
- Generate trading signals
- Create visualizations
- Export results to CSV

#### R Analysis Notebook
```bash
# Open in RStudio or run:
Rscript -e "rmarkdown::render('workbook/exploration.rmd')"
```

The R notebook provides:
- Comprehensive statistical analysis
- Interactive visualizations
- Correlation analysis
- Signal detection algorithms

### Live Dashboard

```bash
streamlit run dashboard.py
```

The dashboard displays:
- Real-time SPY price evolution
- Divergence metrics over time
- Trading signal distribution
- Implied volatility surface
- Recent alerts and events

## Key Metrics

### Divergence Indicators

1. **Price Divergence**: `Call_Price - Put_Price`
2. **IV Divergence**: `Call_IV - Put_IV`
3. **Put-Call Parity Divergence**: Deviation from theoretical parity
4. **Delta Divergence**: `|Call_Delta| - |Put_Delta|`
5. **Momentum Divergence**: Recent vs historical divergence trends

### Signal Generation

- **Z-Score Thresholds**: ±2.0 for price divergence, ±1.5 for IV divergence
- **Rolling Windows**: 10-20 periods for signal calculation
- **Combined Signals**: Majority vote from multiple indicators
- **Extreme Events**: Z-score > 3.0 triggers special alerts

## Configuration

### Divergence Monitor Settings

```python
# In realtime_divergence.py or enhanced_socket.py
monitor = DivergenceMonitor(
    lookback_window=50,      # Number of data points to keep
    signal_threshold=2.0     # Z-score threshold for signals
)
```

### Data Collection Cadence

```python
# In socket scripts
cadence = 10  # Seconds between option data fetches
```

## Example Output

### Console Alerts
```
DIVERGENCE ALERT: BUY | Price Div: 0.1234 | IV Div: 0.0567 | SPY: $623.45
Monitor Status: 45 points | Signals: Buy=3, Sell=1, Neutral=41 | Extreme Events: 2
```

### Generated Files
- `option_data_7_9.csv`: Raw option data with Greeks
- `divergence_alerts_7_9.json`: Real-time alerts
- `atm_divergence_analysis.csv`: Analysis results
- `divergence_data_*.csv`: Exported monitor data
- `divergence_analysis.png`: Visualization plots

## Mathematical Foundation

### Black-Scholes Greeks
- **Delta**: Price sensitivity to underlying movement
- **Gamma**: Delta sensitivity (convexity)
- **Theta**: Time decay
- **Vega**: Volatility sensitivity
- **Rho**: Interest rate sensitivity

### Put-Call Parity
```
C - P = S - K*e^(-r*T)
```
Where:
- C = Call price, P = Put price
- S = Stock price, K = Strike price
- r = Risk-free rate, T = Time to expiry

### Divergence Calculation
```python
def calculate_divergence(call_list, put_list):
    # Multiple metrics including:
    # - Mean divergence
    # - Correlation analysis  
    # - Relative strength
    # - Directional divergence
    # - Momentum divergence
```

## Risk Considerations

- **Market Hours**: System designed for regular trading hours
- **Data Quality**: Filters out invalid/stale option data
- **Signal Lag**: Real-time signals may have 10-30 second delay
- **Backtesting**: Historical analysis doesn't guarantee future performance
- **Position Sizing**: Implement proper risk management

## Troubleshooting

### Common Issues

1. **No Data Loading**
   - Check API credentials in `alpaca_token.py`
   - Verify market hours (9:30 AM - 4:00 PM ET)
   - Ensure data directory exists

2. **Import Errors**
   - Install all requirements: `pip install -r requirements.txt`
   - Check Python path for custom modules

3. **Dashboard Not Loading**
   - Ensure data files exist in `data/` directory
   - Run data collection first
   - Check Streamlit installation

### Debug Mode

```python
# Add to scripts for verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## References

- [Black-Scholes Model](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model)
- [Put-Call Parity](https://en.wikipedia.org/wiki/Put%E2%80%93call_parity)
- [Alpaca API Documentation](https://alpaca.markets/docs/)
- [Options Greeks Explained](https://www.investopedia.com/trading/using-the-greeks-to-understand-options/)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is for educational and research purposes. Use at your own risk for trading decisions.

---

*Built for options traders and quantitative analysts*