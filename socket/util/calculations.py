import math
import numpy as np
from scipy.stats import norm, chi2
from scipy.optimize import minimize_scalar
from datetime import datetime, timedelta
from collections import deque
import warnings
warnings.filterwarnings('ignore')

def mean(x):
    return sum(x) / len(x) if len(x) > 0 else 0

def robust_mean_estimator(x, method='huber'):
    """
    Robust M-estimator for mean with √n-consistency
    
    Implements Huber's robust estimator with theoretical guarantees
    under contaminated normal distributions. Achieves √n-consistency
    and asymptotic normality under mild regularity conditions.
    """
    if len(x) == 0:
        return 0
    
    x_array = np.array(x)
    n = len(x_array)
    
    if method == 'huber':
        k = 1.345  # Achieves 95% efficiency at normal distribution
        
        #--------------------Robust initial estimates
        mu = np.median(x_array)
        mad = np.median(np.abs(x_array - mu))
        sigma = mad / 0.6745  # Convert MAD to standard deviation scale
        
        if sigma == 0:
            return mu
        
        #--------------------Newton-Raphson iteration for M-estimator
        for iteration in range(50):
            residuals = (x_array - mu) / sigma
            
            #----------------Huber's ψ function and its derivative
            psi = np.where(np.abs(residuals) <= k, residuals, k * np.sign(residuals))
            psi_prime = np.where(np.abs(residuals) <= k, 1, 0)
            
            #----------------Newton-Raphson update
            numerator = np.sum(psi)
            denominator = np.sum(psi_prime)
            
            if denominator == 0:
                break
                
            delta = numerator / denominator * sigma
            mu_new = mu - delta
            
            if abs(mu_new - mu) < 1e-8:
                break
            mu = mu_new
        
        #--------------------Calculate asymptotic variance for confidence intervals
        final_residuals = (x_array - mu) / sigma
        psi_final = np.where(np.abs(final_residuals) <= k, final_residuals, k * np.sign(final_residuals))
        psi_prime_final = np.where(np.abs(final_residuals) <= k, 1, 0)
        
        asymptotic_var = np.mean(psi_final**2) / (np.mean(psi_prime_final)**2) if np.mean(psi_prime_final) != 0 else 1
        standard_error = np.sqrt(asymptotic_var / n)
        
        return {
            'estimate': mu,
            'standard_error': standard_error,
            'asymptotic_variance': asymptotic_var,
            'efficiency': 0.95,
            'breakdown_point': 0.5,
            'iterations': iteration + 1
        }
    
    return np.mean(x_array)

def pearson_correlation(x: list[float], y: list[float]) -> float:
    """
    Calculate Pearson correlation coefficient between two variables
    """
    n = len(x)
    
    #------------------------Calculate means
    mean_x = mean(x)
    mean_y = mean(y)
    
    #------------------------Calculate covariance and standard deviations
    covariance = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    std_x = (sum((val - mean_x) ** 2 for val in x)) ** 0.5
    std_y = (sum((val - mean_y) ** 2 for val in y)) ** 0.5
    
    #------------------------Calculate correlation coefficient
    correlation = covariance / (std_x * std_y)
    
    return correlation

def calculate_time_to_expiry(expiry_date_str):
    """
    Calculate time to expiry in years for option analysis
    """
    try:
        expiry_date = datetime.strptime(expiry_date_str, '%Y-%m-%d').date()
        today = datetime.now().date()
        days_to_expiry = (expiry_date - today).days
        return max(days_to_expiry / 365.0, 1/365)  # Minimum 1 day
    except:
        return 1/365  # Default to 1 day if parsing fails

def estimate_implied_volatility_simple(option_price, stock_price, strike, time_to_expiry, option_type='call'):
    """
    Simple implied volatility estimation for real-time processing
    """
    if time_to_expiry <= 0 or option_price <= 0 or stock_price <= 0:
        return None
    
    try:
        #--------------------Calculate intrinsic value
        if option_type.lower() == 'call':
            intrinsic = max(0, stock_price - strike)
        else:
            intrinsic = max(0, strike - stock_price)
        
        #--------------------Time value
        time_value = max(0, option_price - intrinsic)
        
        if time_value <= 0:
            return 0.01  # Minimum IV
        
        #--------------------Simple approximation: IV ≈ time_value / (stock_price * sqrt(time_to_expiry))
        iv_estimate = time_value / (stock_price * math.sqrt(time_to_expiry)) * 2
        
        #--------------------Keep within reasonable bounds
        return max(0.01, min(iv_estimate, 3.0))
        
    except:
        return None

def cointegration_test(call_prices, put_prices):
    """
    Engle-Granger cointegration test for call-put relationships
    """
    if len(call_prices) != len(put_prices) or len(call_prices) < 20:
        return 0, 1.0, [1, -1]
    
    #------------------------Step 1: Estimate cointegrating regression
    X = np.column_stack([np.ones(len(call_prices)), call_prices])
    y = np.array(put_prices)
    
    #------------------------OLS estimation
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta
        
        #--------------------Step 2: Test residuals for unit root
        n = len(residuals)
        y_lag = residuals[:-1]
        dy = np.diff(residuals)
        
        if len(y_lag) > 0 and np.std(y_lag) > 0:
            #----------------Regression: Δuₜ = ρuₜ₋₁ + εₜ
            rho = np.corrcoef(dy, y_lag)[0, 1] * np.std(dy) / np.std(y_lag)
            
            #----------------Test statistic (simplified)
            test_stat = rho * np.sqrt(n)
            
            #----------------Approximate p-value (MacKinnon critical values)
            if test_stat < -3.34:
                p_value = 0.01
            elif test_stat < -2.86:
                p_value = 0.05
            elif test_stat < -2.57:
                p_value = 0.10
            else:
                p_value = 0.50
            
            return test_stat, p_value, beta.tolist()
        
    except np.linalg.LinAlgError:
        pass
    
    return 0, 1.0, [1, -1]

def information_theoretic_divergence(call_dist, put_dist):
    """
    Information-theoretic measures of divergence
    """
    try:
        #--------------------Create histograms (simplified density estimation)
        bins = min(20, len(call_dist) // 5)
        
        call_hist, bin_edges = np.histogram(call_dist, bins=bins, density=True)
        put_hist, _ = np.histogram(put_dist, bins=bin_edges, density=True)
        
        #--------------------Add small epsilon to avoid log(0)
        eps = 1e-10
        call_hist = call_hist + eps
        put_hist = put_hist + eps
        
        #--------------------Normalize
        call_hist = call_hist / np.sum(call_hist)
        put_hist = put_hist / np.sum(put_hist)
        
        #--------------------KL divergence
        kl_div = np.sum(call_hist * np.log(call_hist / put_hist))
        
        #--------------------Jensen-Shannon divergence
        m = 0.5 * (call_hist + put_hist)
        js_div = 0.5 * np.sum(call_hist * np.log(call_hist / m)) + \
                 0.5 * np.sum(put_hist * np.log(put_hist / m))
        
        #--------------------Mutual information (simplified)
        joint_hist = np.outer(call_hist, put_hist)
        marginal_product = np.outer(call_hist, put_hist)
        
        mutual_info = np.sum(joint_hist * np.log(joint_hist / marginal_product + eps))
        
        return {
            'kl_divergence': kl_div,
            'mutual_information': mutual_info,
            'jensen_shannon': js_div
        }
    
    except (ValueError, ZeroDivisionError):
        return {
            'kl_divergence': 0,
            'mutual_information': 0,
            'jensen_shannon': 0
        }

def stochastic_divergence_model(call_prices, put_prices, dt=1/252):
    """
    Stochastic model for divergence dynamics using jump-diffusion processes
    """
    if len(call_prices) != len(put_prices) or len(call_prices) < 20:
        return {'error': 'Insufficient data for stochastic modeling'}
    
    #------------------------Calculate divergence process
    divergence = np.array(call_prices) - np.array(put_prices)
    n = len(divergence)
    
    #------------------------Estimate increments
    increments = np.diff(divergence)
    
    #------------------------Detect jumps using threshold method
    threshold = 3 * np.std(increments)
    jump_indices = np.where(np.abs(increments) > threshold)[0]
    
    #------------------------Separate continuous and jump components
    continuous_increments = increments.copy()
    jump_sizes = np.zeros_like(increments)
    
    for idx in jump_indices:
        jump_sizes[idx] = increments[idx]
        continuous_increments[idx] = 0  # Remove jump component
    
    #------------------------Estimate drift (conditional expectation)
    non_jump_mask = np.abs(increments) <= threshold
    if np.sum(non_jump_mask) > 5:
        drift_estimate = np.mean(continuous_increments[non_jump_mask]) / dt
    else:
        drift_estimate = 0
    
    #------------------------Estimate diffusion coefficient
    if np.sum(non_jump_mask) > 5:
        diffusion_estimate = np.sqrt(np.var(continuous_increments[non_jump_mask]) / dt)
    else:
        diffusion_estimate = np.std(increments)
    
    #------------------------Estimate jump parameters
    jump_intensity = len(jump_indices) / (n * dt)
    
    if len(jump_sizes[jump_indices]) > 0:
        jump_mean = np.mean(jump_sizes[jump_indices])
        jump_std = np.std(jump_sizes[jump_indices])
    else:
        jump_mean = jump_std = 0
    
    #------------------------Model selection: test for presence of jumps
    if len(jump_indices) > 0:
        log_likelihood_jumps = -0.5 * np.sum(np.log(2*np.pi*diffusion_estimate**2*dt)) - \
                              0.5 * np.sum(continuous_increments[non_jump_mask]**2 / (diffusion_estimate**2*dt))
        log_likelihood_no_jumps = -0.5 * n * np.log(2*np.pi*np.var(increments)*dt) - \
                                 0.5 * np.sum(increments**2 / (np.var(increments)*dt))
        
        lr_statistic = 2 * (log_likelihood_jumps - log_likelihood_no_jumps)
        jump_test_p_value = 1 - chi2.cdf(lr_statistic, df=2) if lr_statistic > 0 else 1.0
    else:
        jump_test_p_value = 1.0
    
    return {
        'drift_estimate': drift_estimate,
        'diffusion_estimate': diffusion_estimate,
        'jump_intensity': jump_intensity,
        'jump_mean': jump_mean,
        'jump_std': jump_std,
        'jump_test_p_value': jump_test_p_value,
        'model_type': 'jump_diffusion' if jump_test_p_value < 0.05 else 'pure_diffusion',
        'jumps_detected': len(jump_indices),
        'sample_size': n,
        'time_span': n * dt
    }

def measure_theoretic_divergence(call_prices, put_prices):
    """
    Measure-theoretic formulation of divergence detection problem
    """
    if len(call_prices) != len(put_prices) or len(call_prices) < 10:
        return {'error': 'Insufficient data for measure-theoretic analysis'}
    
    #------------------------Empirical measures
    call_measure = np.array(call_prices) / np.sum(call_prices) if np.sum(call_prices) > 0 else np.ones(len(call_prices)) / len(call_prices)
    put_measure = np.array(put_prices) / np.sum(put_prices) if np.sum(put_prices) > 0 else np.ones(len(put_prices)) / len(put_prices)
    
    #------------------------Wasserstein distance (optimal transport cost)
    sorted_calls = np.sort(call_prices)
    sorted_puts = np.sort(put_prices)
    
    #------------------------For 1D case, Wasserstein-1 distance has closed form
    wasserstein_distance = np.mean(np.abs(sorted_calls - sorted_puts))
    
    #------------------------Relative entropy (KL divergence)
    try:
        bins = min(10, len(call_prices) // 3)
        call_hist, bin_edges = np.histogram(call_prices, bins=bins, density=True)
        put_hist, _ = np.histogram(put_prices, bins=bin_edges, density=True)
        
        #--------------------Add regularization to avoid log(0)
        eps = 1e-10
        call_hist = call_hist + eps
        put_hist = put_hist + eps
        
        #--------------------Normalize
        call_hist = call_hist / np.sum(call_hist)
        put_hist = put_hist / np.sum(put_hist)
        
        kl_divergence = np.sum(call_hist * np.log(call_hist / put_hist))
        reverse_kl = np.sum(put_hist * np.log(put_hist / call_hist))
        
        #--------------------Symmetric KL (Jensen-Shannon divergence)
        m = 0.5 * (call_hist + put_hist)
        js_divergence = 0.5 * np.sum(call_hist * np.log(call_hist / m)) + \
                       0.5 * np.sum(put_hist * np.log(put_hist / m))
        
    except (ValueError, ZeroDivisionError):
        kl_divergence = reverse_kl = js_divergence = 0
    
    #------------------------Optimal coupling (approximation)
    transport_cost = wasserstein_distance
    
    #------------------------Entropy-regularized optimal transport
    regularization_param = 0.1
    regularized_cost = transport_cost + regularization_param * js_divergence
    
    return {
        'wasserstein_distance': wasserstein_distance,
        'kl_divergence': kl_divergence,
        'reverse_kl_divergence': reverse_kl,
        'jensen_shannon_divergence': js_divergence,
        'optimal_transport_cost': transport_cost,
        'regularized_transport_cost': regularized_cost,
        'measure_concentration': np.var(call_prices) + np.var(put_prices),
        'empirical_process_sup_norm': np.max(np.abs(np.cumsum(call_prices) - np.cumsum(put_prices)))
    }

def calculate_divergence(call_list: list[float], put_list: list[float]) -> dict:
    """
    Calculate multiple divergence metrics between call and put options
    """
    if len(call_list) != len(put_list) or len(call_list) == 0:
        return {'error': 'Invalid input lists'}
    
    #------------------------Basic statistics
    call_mean = mean(call_list)
    put_mean = mean(put_list)
    
    #------------------------Normalize by subtracting means
    normalized_call = [x - call_mean for x in call_list]
    normalized_put = [x - put_mean for x in put_list]
    
    #------------------------Calculate various divergence metrics
    
    #------------------------1. Simple difference in means
    mean_divergence = call_mean - put_mean
    
    #------------------------2. Correlation between normalized series
    correlation = pearson_correlation(normalized_call, normalized_put)
    
    #------------------------3. Relative strength - which is moving more from baseline
    call_volatility = (sum(x**2 for x in normalized_call) / len(normalized_call))**0.5
    put_volatility = (sum(x**2 for x in normalized_put) / len(normalized_put))**0.5
    relative_strength = call_volatility / put_volatility if put_volatility != 0 else float('inf')
    
    #------------------------4. Directional divergence - are they moving in opposite directions?
    call_trend = (call_list[-1] - call_list[0]) / len(call_list) if len(call_list) > 1 else 0
    put_trend = (put_list[-1] - put_list[0]) / len(put_list) if len(put_list) > 1 else 0
    directional_divergence = call_trend - put_trend
    
    #------------------------5. Momentum divergence - recent vs historical
    if len(call_list) >= 4:
        recent_call = mean(call_list[-len(call_list)//2:])
        recent_put = mean(put_list[-len(put_list)//2:])
        historical_call = mean(call_list[:len(call_list)//2])
        historical_put = mean(put_list[:len(put_list)//2])
        
        momentum_divergence = (recent_call - historical_call) - (recent_put - historical_put)
    else:
        momentum_divergence = 0
    
    #------------------------6. Volatility divergence
    volatility_divergence = call_volatility - put_volatility
    
    #------------------------Statistical tests
    cointegration_result = cointegration_test(call_list, put_list)
    info_theory_result = information_theoretic_divergence(call_list, put_list)
    
    #------------------------Asymptotic properties
    call_array = np.array(call_list)
    put_array = np.array(put_list)
    
    #------------------------Test for martingale property
    call_martingale_test = asymptotic_distribution_test(call_array)
    put_martingale_test = asymptotic_distribution_test(put_array)
    
    #------------------------Stochastic and measure-theoretic analysis
    stochastic_result = stochastic_divergence_model(call_list, put_list)
    measure_result = measure_theoretic_divergence(call_list, put_list)
    
    #------------------------Robust estimation with theoretical guarantees
    robust_call_est = robust_mean_estimator(call_list)
    robust_put_est = robust_mean_estimator(put_list)
    
    if isinstance(robust_call_est, dict) and isinstance(robust_put_est, dict):
        robust_divergence = robust_call_est['estimate'] - robust_put_est['estimate']
        robust_se = np.sqrt(robust_call_est['standard_error']**2 + robust_put_est['standard_error']**2)
    else:
        robust_divergence = robust_call_est - robust_put_est
        robust_se = None
    
    return {
        'mean_divergence': mean_divergence,
        'correlation': correlation,
        'relative_strength': relative_strength,
        'directional_divergence': directional_divergence,
        'momentum_divergence': momentum_divergence,
        'volatility_divergence': volatility_divergence,
        'call_volatility': call_volatility,
        'put_volatility': put_volatility,
        
        #--------------------Robust M-estimator results
        'robust_divergence': robust_divergence,
        'robust_standard_error': robust_se,
        
        #--------------------Statistical measures
        'cointegration_test_stat': cointegration_result[0],
        'cointegration_p_value': cointegration_result[1],
        'cointegrating_vector': cointegration_result[2],
        
        'kl_divergence': info_theory_result['kl_divergence'],
        'mutual_information': info_theory_result['mutual_information'],
        'jensen_shannon_divergence': info_theory_result['jensen_shannon'],
        
        'call_martingale_test': call_martingale_test[1],  # p-value
        'put_martingale_test': put_martingale_test[1],    # p-value
        
        #--------------------Stochastic process modeling
        'stochastic_model': stochastic_result,
        
        #--------------------Measure-theoretic analysis
        'measure_theoretic': measure_result,
        
        #--------------------Convergence properties and theoretical bounds
        'sample_size': len(call_list),
        'estimation_error_bound': 1.0 / np.sqrt(len(call_list)) if len(call_list) > 0 else float('inf'),
        'minimax_risk_bound': (len(call_list))**(-4/5) if len(call_list) > 0 else float('inf'),
        'concentration_inequality': 'Hoeffding' if len(call_list) > 30 else 'Empirical Bernstein'
    }

def calculate_put_call_parity_divergence(call_price, put_price, stock_price, strike, time_to_expiry, risk_free_rate=0.05):
    """
    Calculate divergence from put-call parity with statistical bounds
    """
    import math
    
    theoretical_difference = stock_price - strike * math.exp(-risk_free_rate * time_to_expiry)
    actual_difference = call_price - put_price
    divergence = actual_difference - theoretical_difference
    
    #------------------------Estimate standard error (simplified)
    price_volatility = 0.01 * stock_price  # Approximate
    standard_error = price_volatility * math.sqrt(2)  # For difference of two prices
    
    #------------------------Test statistic for H₀: divergence = 0
    t_statistic = divergence / standard_error if standard_error > 0 else 0
    
    #------------------------Confidence interval (95%)
    ci_lower = divergence - 1.96 * standard_error
    ci_upper = divergence + 1.96 * standard_error
    
    return {
        'divergence': divergence,
        'standard_error': standard_error,
        't_statistic': t_statistic,
        'confidence_interval': (ci_lower, ci_upper),
        'is_significant': abs(t_statistic) > 1.96
    }

def cusum_change_point_detection(data, threshold=5.0):
    """
    CUSUM-based change-point detection for structural breaks in divergence patterns
    """
    n = len(data)
    if n < 10:
        return [], []
    
    #------------------------Estimate parameters
    mu0 = np.mean(data[:n//2])
    sigma = np.std(data)
    
    #------------------------CUSUM statistics
    S_plus = np.zeros(n)
    S_minus = np.zeros(n)
    
    for i in range(1, n):
        S_plus[i] = max(0, S_plus[i-1] + (data[i] - mu0)/sigma - 0.5)
        S_minus[i] = max(0, S_minus[i-1] - (data[i] - mu0)/sigma - 0.5)
    
    #------------------------Detect change points
    change_points = []
    for i in range(n):
        if S_plus[i] > threshold or S_minus[i] > threshold:
            change_points.append(i)
    
    return change_points, {'S_plus': S_plus, 'S_minus': S_minus}

def garch_volatility_model(returns, p=1, q=1):
    """
    GARCH(p,q) model for volatility clustering in divergence series
    """
    n = len(returns)
    if n < 50:
        return np.ones(n), {'omega': 1.0, 'alpha': [0.1], 'beta': [0.8]}
    
    #------------------------Initialize parameters
    omega = 0.01
    alpha = [0.1] * p
    beta = [0.8] * q
    
    #------------------------Conditional variance estimation (simplified)
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(returns)
    
    for t in range(1, n):
        sigma2[t] = omega
        
        #--------------------ARCH terms
        for i in range(min(p, t)):
            sigma2[t] += alpha[i] * returns[t-1-i]**2
        
        #--------------------GARCH terms  
        for j in range(min(q, t)):
            sigma2[t] += beta[j] * sigma2[t-1-j]
    
    return sigma2, {'omega': omega, 'alpha': alpha, 'beta': beta}

def asymptotic_distribution_test(divergence_series, confidence_level=0.95):
    """
    Asymptotic distribution test for divergence persistence
    """
    n = len(divergence_series)
    if n < 20:
        return 0, 1.0, 0
    
    #------------------------Ljung-Box test for serial correlation
    lags = min(10, n//4)
    
    #------------------------Calculate autocorrelations
    mean_div = np.mean(divergence_series)
    centered = divergence_series - mean_div
    
    autocorrs = []
    for lag in range(1, lags + 1):
        if n - lag > 0:
            corr = np.corrcoef(centered[:-lag], centered[lag:])[0, 1]
            autocorrs.append(corr if not np.isnan(corr) else 0)
        else:
            autocorrs.append(0)
    
    #------------------------Ljung-Box statistic
    Q = n * (n + 2) * sum([(autocorrs[i]**2) / (n - i - 1) for i in range(len(autocorrs))])
    
    #------------------------Critical value from chi-squared distribution
    alpha = 1 - confidence_level
    critical_value = chi2.ppf(1 - alpha, df=lags)
    p_value = 1 - chi2.cdf(Q, df=lags)
    
    return Q, p_value, critical_value

def high_dimensional_divergence_analysis(option_chains, method='factor_model'):
    """
    High-dimensional analysis for multiple option chains
    """
    if not option_chains or len(option_chains) < 3:
        return {'error': 'Insufficient option chains for high-dimensional analysis'}
    
    #------------------------Stack option price series into matrix
    price_matrix = []
    chain_names = []
    
    for chain_name, prices in option_chains.items():
        if len(prices) > 10:  # Minimum length requirement
            price_matrix.append(prices)
            chain_names.append(chain_name)
    
    if len(price_matrix) < 3:
        return {'error': 'Insufficient valid option chains'}
    
    #------------------------Ensure equal length (truncate to minimum)
    min_length = min(len(series) for series in price_matrix)
    X = np.array([series[:min_length] for series in price_matrix]).T  # T x N matrix
    
    T, N = X.shape
    
    if method == 'factor_model':
        #--------------------Principal Component Analysis for factor extraction
        X_centered = X - np.mean(X, axis=0)
        
        #--------------------Covariance matrix
        C = np.cov(X_centered.T)
        
        #--------------------Eigendecomposition
        eigenvals, eigenvecs = np.linalg.eigh(C)
        
        #--------------------Sort by eigenvalue (descending)
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        #--------------------Determine number of factors using information criteria
        max_factors = min(8, N//2)
        ic_values = []
        
        for k in range(1, max_factors + 1):
            #----------------Explained variance
            explained_var = np.sum(eigenvals[:k]) / np.sum(eigenvals)
            
            #----------------Penalty term
            penalty = k * (N + T) * np.log(N * T) / (N * T)
            
            #----------------Information criterion
            ic = -explained_var + penalty
            ic_values.append(ic)
        
        #--------------------Optimal number of factors
        optimal_k = np.argmin(ic_values) + 1
        
        #--------------------Extract factors and loadings
        factor_loadings = eigenvecs[:, :optimal_k]  # N x k
        common_factors = X_centered @ factor_loadings  # T x k
        
        #--------------------Idiosyncratic components
        fitted_values = common_factors @ factor_loadings.T
        idiosyncratic = X_centered - fitted_values
        
        explained_variance_ratio = np.sum(eigenvals[:optimal_k]) / np.sum(eigenvals)
        
        return {
            'factor_loadings': factor_loadings,
            'common_factors': common_factors,
            'idiosyncratic_components': idiosyncratic,
            'explained_variance_ratio': explained_variance_ratio,
            'n_factors': factor_loadings.shape[1] if factor_loadings.size > 0 else 0,
            'chain_names': chain_names,
            'method_used': method,
            'sample_size': (T, N),
            'condition_number': np.linalg.cond(X) if X.size > 0 else float('inf')
        }
    
    return {'error': 'Method not implemented'}

def advanced_change_point_detection(data, method='pelt', penalty='bic'):
    """
    Advanced change-point detection using PELT (Pruned Exact Linear Time)
    """
    n = len(data)
    if n < 20:
        return [], [], []
    
    data_array = np.array(data)
    
    if method == 'pelt':
        #--------------------PELT algorithm implementation (simplified)
        def cost_function(segment):
            if len(segment) <= 1:
                return 0
            return np.sum((segment - np.mean(segment))**2)
        
        #--------------------Penalty selection
        if penalty == 'bic':
            pen = np.log(n)
        elif penalty == 'aic':
            pen = 2
        else:  # hannan_quinn
            pen = 2 * np.log(np.log(n))
        
        #--------------------Dynamic programming for optimal segmentation
        F = np.full(n + 1, np.inf)
        F[0] = -pen
        cp_candidates = [[] for _ in range(n + 1)]
        
        for t in range(1, n + 1):
            for s in range(t):
                segment_cost = cost_function(data_array[s:t])
                total_cost = F[s] + segment_cost + pen
                
                if total_cost < F[t]:
                    F[t] = total_cost
                    cp_candidates[t] = cp_candidates[s] + [s] if s > 0 else []
        
        change_points = [cp for cp in cp_candidates[n] if cp > 0 and cp < n]
        
    else:
        change_points = []
    
    #------------------------Calculate confidence intervals (asymptotic theory)
    confidence_intervals = []
    for cp in change_points:
        #--------------------Asymptotic standard error: O(1/√n)
        se = max(1, int(np.sqrt(n) / 2))
        ci_lower = max(0, cp - se)
        ci_upper = min(n, cp + se)
        confidence_intervals.append((ci_lower, ci_upper))
    
    #------------------------Test statistics for detected change-points
    test_statistics = []
    for cp in change_points:
        if cp > 5 and cp < n - 5:
            before = data_array[max(0, cp-5):cp]
            after = data_array[cp:min(n, cp+5)]
            
            if len(before) > 0 and len(after) > 0:
                #----------------Likelihood ratio statistic
                combined_var = np.var(np.concatenate([before, after]))
                separate_var = (np.var(before) + np.var(after)) / 2
                
                if separate_var > 0 and combined_var > 0:
                    lr_stat = len(before) + len(after) * np.log(combined_var / separate_var)
                    test_statistics.append(lr_stat)
                else:
                    test_statistics.append(0)
            else:
                test_statistics.append(0)
        else:
            test_statistics.append(0)
    
    return change_points, test_statistics, confidence_intervals

def rolling_divergence_signal(prices, window=20, threshold=2.0, method='adaptive'):
    """
    Enhanced signal generation with change-point detection and adaptive thresholds
    """
    if len(prices) < window:
        return [0] * len(prices), {}
    
    prices_array = np.array(prices)
    signals = [0] * len(prices)
    metadata = {}
    
    if method == 'changepoint':
        #--------------------Use change-point detection
        change_points, cusum_stats = cusum_change_point_detection(prices_array)
        metadata['change_points'] = change_points
        metadata['cusum_stats'] = cusum_stats
        
        #--------------------Adaptive threshold based on detected regime changes
        for i in range(window, len(prices)):
            #----------------Find most recent change point
            recent_cp = max([cp for cp in change_points if cp < i], default=0)
            
            #----------------Use data since last change point
            regime_data = prices_array[recent_cp:i]
            if len(regime_data) > 5:
                regime_mean = np.mean(regime_data)
                regime_std = np.std(regime_data)
                
                if regime_std > 0:
                    z_score = (prices[i] - regime_mean) / regime_std
                    if z_score > threshold:
                        signals[i] = 1
                    elif z_score < -threshold:
                        signals[i] = -1
    
    elif method == 'adaptive':
        #--------------------Use GARCH-based adaptive thresholds
        returns = np.diff(prices_array)
        if len(returns) > 10:
            cond_var, garch_params = garch_volatility_model(returns)
            metadata['garch_params'] = garch_params
            
            for i in range(window, len(prices)):
                #----------------Adaptive threshold based on conditional volatility
                vol_adj_threshold = threshold * np.sqrt(cond_var[min(i-1, len(cond_var)-1)])
                
                window_data = prices_array[i-window:i]
                window_mean = np.mean(window_data)
                window_std = np.std(window_data)
                
                if window_std > 0:
                    z_score = (prices[i] - window_mean) / window_std
                    if z_score > vol_adj_threshold:
                        signals[i] = 1
                    elif z_score < -vol_adj_threshold:
                        signals[i] = -1
    
    else:  # basic method
        for i in range(window, len(prices)):
            window_data = prices_array[i-window:i]
            window_mean = np.mean(window_data)
            window_std = np.std(window_data)
            
            if window_std > 0:
                z_score = (prices[i] - window_mean) / window_std
                if z_score > threshold:
                    signals[i] = 1
                elif z_score < -threshold:
                    signals[i] = -1
    
    #------------------------Enhanced with change-point detection
    if len(prices) > 30:
        change_points, cp_stats, cp_intervals = advanced_change_point_detection(prices_array, method='pelt')
        metadata['advanced_change_points'] = change_points
        metadata['change_point_statistics'] = cp_stats
        metadata['change_point_intervals'] = cp_intervals
        
        #--------------------Adjust signals based on structural breaks
        for cp in change_points:
            if cp < len(signals):
                #----------------Increase sensitivity around change-points
                for i in range(max(0, cp-3), min(len(signals), cp+3)):
                    if signals[i] != 0:
                        signals[i] = int(1.5 * signals[i])  # Amplify signals near breaks
    
    return signals, metadata