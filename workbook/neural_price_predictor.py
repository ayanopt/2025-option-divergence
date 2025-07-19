import pandas as pd
import numpy as np
import torch
from torch import nn
from pathlib import Path
from scipy import stats
import math

# -------------------------------------------------------------
# Configuration
OPTION_COUNT = 15
FEATURES = [
    'implied_volatility', 'delta', 'gamma', 'theta',
    'vega', 'rho', 'moneyness'
]

# -------------------------------------------------------------
def load_option_matrix(directory=None, option_count=OPTION_COUNT, future_period=15):
    """Load all processed CSVs and convert to feature matrices."""
    if directory is None:
        directory = Path(__file__).resolve().parent.parent / 'data' / 'processed'
    else:
        directory = Path(directory)

    files = list(directory.glob('option_data_with_future_prices_*.csv'))
    if not files:
        raise FileNotFoundError(f"No option CSVs found in {directory}")

    dfs = []
    for path in files:
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)

    grouped = data.groupby('timestamp')

    X_list, y_list, iv_list, diff_list = [], [], [], []

    for ts, group in grouped:
        calls = group[group['option_type'] == 'call'].sort_values('strike')
        puts = group[group['option_type'] == 'put'].sort_values('strike')

        feat_vec = []
        price_vec = []
        iv_vec = []
        diff_vec = []

        for subset in (calls, puts):
            rows = subset.head(option_count)
            for _, row in rows.iterrows():
                feat_vec.extend(row[FEATURES].to_list())
                price_vec.append(row['standardized_price'])
                iv_vec.append(row['implied_volatility'])
                diff_key = f'price_diff_{future_period}_periods'
                diff_val = row.get(diff_key, 0.0)
                diff_vec.append(diff_val)
            if len(rows) < option_count:
                pad_count = option_count - len(rows)
                feat_vec.extend([0.0] * pad_count * len(FEATURES))
                price_vec.extend([0.0] * pad_count)
                iv_vec.extend([1.0] * pad_count)
                diff_vec.extend([0.0] * pad_count)

        X_list.append(feat_vec)
        y_list.append(price_vec)
        iv_list.append(iv_vec)
        diff_list.append(diff_vec)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    iv = np.array(iv_list, dtype=np.float32)
    diffs = np.array(diff_list, dtype=np.float32)
    return X, y, iv, diffs

# -------------------------------------------------------------
class PriceNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# -------------------------------------------------------------
def train_model(X_train, y_train, X_val, y_val, epochs: int = 200, lr: float = 1e-3, patience: int = 20):
    X_train = torch.tensor(X_train)
    y_train = torch.tensor(y_train)
    X_val = torch.tensor(X_val)
    y_val = torch.tensor(y_val)

    model = PriceNet(X_train.shape[1], y_train.shape[1])
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_loss = float('inf')
    best_state = None
    counter = 0
    for epoch in range(epochs):
        model.train()
        optim.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        optim.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = loss_fn(val_pred, y_val).item()

        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: train={loss.item():.6f} val={val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# -------------------------------------------------------------
def bayesian_deviation_probability(pred, actual, iv, sigma):
    """Posterior probability of a deviation using the standardized price as prior."""
    var_prior = np.maximum(iv, 1e-6) ** 2
    var_like = sigma ** 2
    var_post = 1.0 / (1.0 / var_prior + 1.0 / var_like)
    mean_post = var_post * (actual / var_prior + pred / var_like)
    diff = np.abs(mean_post - actual)
    return 2 * (1 - stats.norm.cdf(diff, loc=0.0, scale=np.sqrt(var_post)))

# -------------------------------------------------------------
def train_test_split(X, y, iv, diffs, test_ratio=0.2):
    n = len(X)
    idx = int(n * (1 - test_ratio))
    return (X[:idx], y[:idx], iv[:idx], diffs[:idx]), (X[idx:], y[idx:], iv[idx:], diffs[idx:])

# -------------------------------------------------------------
def simulate_trading(model, sigma, X, y, diffs, threshold=0.05):
    profits = []
    with torch.no_grad():
        preds = model(torch.tensor(X, dtype=torch.float32)).numpy()
    for i in range(len(X)):
        call_slice = slice(0, OPTION_COUNT)
        put_slice = slice(OPTION_COUNT, OPTION_COUNT * 2)
        call_pred = preds[i, call_slice]
        put_pred = preds[i, put_slice]
        call_actual = y[i, call_slice]
        put_actual = y[i, put_slice]
        call_diff = call_pred - call_actual
        put_diff = put_pred - put_actual
        call_prob = bayesian_deviation_probability(call_pred, call_actual,
                                                   iv[i, :OPTION_COUNT], sigma)
        put_prob = bayesian_deviation_probability(put_pred, put_actual,
                                                  iv[i, OPTION_COUNT:], sigma)

        call_candidates = np.where((call_diff > 0) & (call_prob < threshold))[0]
        put_candidates = np.where((put_diff > 0) & (put_prob < threshold))[0]
        if len(call_candidates) == 0 or len(put_candidates) == 0:
            continue
        cidx = call_candidates[np.argmax(call_diff[call_candidates])]
        pidx = put_candidates[np.argmax(put_diff[put_candidates])]
        profit = diffs[i, cidx] + diffs[i, pidx]
        profits.append(profit)
    return np.array(profits)

# -------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--period", type=int, default=15,
                        help="future horizon for price difference")
    args = parser.parse_args()

    X, y, iv, diffs = load_option_matrix(future_period=args.period)
    (train_X, train_y, train_iv, train_diffs), (test_X, test_y, test_iv, test_diffs) = train_test_split(X, y, iv, diffs)
    # split training set for validation
    (tr_X, tr_y, tr_iv, tr_diffs), (val_X, val_y, val_iv, val_diffs) = train_test_split(train_X, train_y, train_iv, train_diffs, test_ratio=0.2)

    model = train_model(tr_X, tr_y, val_X, val_y)

    with torch.no_grad():
        val_pred = model(torch.tensor(val_X, dtype=torch.float32))
    sigma = (val_pred.numpy() - val_y).std()
    trade_profits = simulate_trading(model, sigma, test_X, test_y, test_diffs)
    if len(trade_profits) > 0:
        print("Mean profit:", trade_profits.mean())
    else:
        print("No trades executed")
