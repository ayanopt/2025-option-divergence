import math
from scipy.stats import norm
from datetime import datetime, timedelta

def mean(x):
    return sum(x)/ len(x)

def pearson_correlation(x: list[float], y: list[float]) -> float:
    """
    Calculate Pearson correlation coefficient between two variables
    
    Args:
        x: First variable (list or array of numbers)
        y: Second variable (list or array of numbers) 
        
    Returns:
        Correlation coefficient between -1 and 1
    """
    n = len(x)
    
    # Calculate means
    mean_x = mean(x)
    mean_y = mean(y)
    
    # Calculate covariance and standard deviations
    covariance = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    std_x = (sum((val - mean_x) ** 2 for val in x)) ** 0.5
    std_y = (sum((val - mean_y) ** 2 for val in y)) ** 0.5
    
    # Calculate correlation coefficient
    correlation = covariance / (std_x * std_y)
    
    return correlation

def calculate_divergence(call_list: list[float], put_list: list[float]) -> dict:
    """
    Calculate multiple divergence metrics between call and put options
    Args:
        call_list: list of call option prices/metrics
        put_list: list of put option prices/metrics
        
    Returns:
        Dictionary containing various divergence metrics
    """
    if len(call_list) != len(put_list) or len(call_list) == 0:
        return {'error': 'Invalid input lists'}
    
    # Basic statistics
    call_mean = mean(call_list)
    put_mean = mean(put_list)
    
    # Normalize by subtracting means
    normalized_call = [x - call_mean for x in call_list]
    normalized_put = [x - put_mean for x in put_list]
    
    # Calculate various divergence metrics
    
    # 1. Simple difference in means
    mean_divergence = call_mean - put_mean
    
    # 2. Correlation between normalized series
    correlation = pearson_correlation(normalized_call, normalized_put)
    
    # 3. Relative strength - which is moving more from baseline
    call_volatility = (sum(x**2 for x in normalized_call) / len(normalized_call))**0.5
    put_volatility = (sum(x**2 for x in normalized_put) / len(normalized_put))**0.5
    relative_strength = call_volatility / put_volatility if put_volatility != 0 else float('inf')
    
    # 4. Directional divergence - are they moving in opposite directions?
    call_trend = (call_list[-1] - call_list[0]) / len(call_list) if len(call_list) > 1 else 0
    put_trend = (put_list[-1] - put_list[0]) / len(put_list) if len(put_list) > 1 else 0
    directional_divergence = call_trend - put_trend
    
    # 5. Momentum divergence - recent vs historical
    if len(call_list) >= 4:
        recent_call = mean(call_list[-len(call_list)//2:])
        recent_put = mean(put_list[-len(put_list)//2:])
        historical_call = mean(call_list[:len(call_list)//2])
        historical_put = mean(put_list[:len(put_list)//2])
        
        momentum_divergence = (recent_call - historical_call) - (recent_put - historical_put)
    else:
        momentum_divergence = 0
    
    # 6. Volatility divergence
    volatility_divergence = call_volatility - put_volatility
    
    return {
        'mean_divergence': mean_divergence,
        'correlation': correlation,
        'relative_strength': relative_strength,
        'directional_divergence': directional_divergence,
        'momentum_divergence': momentum_divergence,
        'volatility_divergence': volatility_divergence,
        'call_volatility': call_volatility,
        'put_volatility': put_volatility
    }

def black_scholes_greeks(S, K, T, r, sigma, option_type='call'):
    """Calculate Black-Scholes greeks"""
    if T <= 0 or sigma <= 0:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
    
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    if option_type == 'call':
        delta = norm.cdf(d1)
        rho = K * T * math.exp(-r * T) * norm.cdf(d2) / 100
        theta = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        delta = norm.cdf(d1) - 1
        rho = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100
        theta = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365
    
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * norm.pdf(d1) * math.sqrt(T) / 100
    
    return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega, 'rho': rho}

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """Calculate Black-Scholes option price"""
    if T <= 0 or sigma <= 0:
        return 0
    
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    if option_type == 'call':
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def implied_volatility(S, K, T, r, market_price, option_type='call'):
    """Calculate implied volatility using Newton-Raphson"""
    if T <= 0 or market_price <= 0:
        return None
    
    sigma = 0.2  # Initial guess
    for _ in range(100):
        price = black_scholes_price(S, K, T, r, sigma, option_type)
        vega = S * norm.pdf((math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))) * math.sqrt(T)
        
        if abs(vega) < 1e-6:
            break
            
        sigma_new = sigma - (price - market_price) / vega
        if abs(sigma_new - sigma) < 1e-6:
            return sigma_new
        sigma = max(sigma_new, 0.01)
    # TODO: improve sub zero sigma handling
    return sigma if sigma > 0 else None

def get_time_to_expiry(expiry_date):
    """Calculate time to expiry in years"""
    from datetime import datetime, date
    if isinstance(expiry_date, str):
        expiry = datetime.strptime(expiry_date[:8], '%Y%m%d').date()
    else:
        expiry = expiry_date
    return max((expiry - date.today()).days / 365.0, 1/365)

def calculate_put_call_parity_divergence(call_price, put_price, stock_price, strike, time_to_expiry, risk_free_rate=0.05):
    """
    Calculate divergence from put-call parity
    Put-Call Parity: C - P = S - K*e^(-r*T)
    
    Args:
        call_price: Current call option price
        put_price: Current put option price  
        stock_price: Current stock price
        strike: Strike price
        time_to_expiry: Time to expiry in years
        risk_free_rate: Risk-free interest rate
        
    Returns:
        Divergence from theoretical parity value
    """
    import math
    
    theoretical_difference = stock_price - strike * math.exp(-risk_free_rate * time_to_expiry)
    actual_difference = call_price - put_price
    
    return actual_difference - theoretical_difference

def rolling_divergence_signal(prices, window=20, threshold=2.0):
    """
    Generate divergence signals based on rolling statistics
    
    Args:
        prices: List of price differences or divergence values
        window: Rolling window size
        threshold: Z-score threshold for signal generation
        
    Returns:
        List of signals (-1, 0, 1) for sell, neutral, buy
    """
    if len(prices) < window:
        return [0] * len(prices)
    
    signals = [0] * (window - 1)  # No signals for first window-1 points
    
    for i in range(window - 1, len(prices)):
        window_data = prices[i - window + 1:i + 1]
        window_mean = mean(window_data)
        window_std = (sum((x - window_mean)**2 for x in window_data) / len(window_data))**0.5
        
        if window_std == 0:
            signals.append(0)
            continue
            
        z_score = (prices[i] - window_mean) / window_std
        
        if z_score > threshold:
            signals.append(1)  # Buy signal
        elif z_score < -threshold:
            signals.append(-1)  # Sell signal
        else:
            signals.append(0)  # Neutral
    
    return signals