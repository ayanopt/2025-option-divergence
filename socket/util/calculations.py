import math
from scipy.stats import norm

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
    
    #-------------Calculate means
    mean_x = mean(x)
    mean_y = mean(y)
    
    #-------------Calculate covariance and standard deviations
    covariance = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    std_x = (sum((val - mean_x) ** 2 for val in x)) ** 0.5
    std_y = (sum((val - mean_y) ** 2 for val in y)) ** 0.5
    
    #-------------Calculate correlation coefficient
    correlation = covariance / (std_x * std_y)
    
    return correlation

def calculate_divergence(call_list: list[float], put_list: list[float]) -> float:
    """
    Subtract mean from each field -> Measure deviation from mean between each
    Args:
        x: list of call price
        y: list of put price
        
    Returns:
        Which derivative is trending more than the other
    """

    normalized_call = [x - mean(call_list) for x in call_list]
    normalized_put = [x - mean(put_list) for x in put_list]
    pass
    #-------------TODO: implement

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
    #-------------TODO: improve sub zero sigma handling
    return sigma if sigma > 0 else None

def get_time_to_expiry(expiry_date):
    """Calculate time to expiry in years"""
    from datetime import datetime, date
    if isinstance(expiry_date, str):
        expiry = datetime.strptime(expiry_date[:8], '%Y%m%d').date()
    else:
        expiry = expiry_date
    return max((expiry - date.today()).days / 365.0, 1/365)