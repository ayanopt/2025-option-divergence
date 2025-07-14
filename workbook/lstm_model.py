import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random,math
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import TimeDistributed, Flatten, Reshape, BatchNormalization, Attention
from tensorflow.keras.regularizers import l2
from joblib import dump
from analyze_processed_data import apply_standardization
#-----------------------------------------------------------------------------
def load_and_group_train_data():
    data_dir = Path("../data/processed")
    
    #------------Load all processed data files
    all_data = []
    for file in data_dir.glob("option_data_with_future_prices_7_8*.csv"):
        df = pd.read_csv(file)
        all_data.append(df)
    
    #------------Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    #------------Apply log transformation to price_diff columns
    price_diff_cols = [col for col in combined_df.columns if col.startswith('price_diff_') and col.endswith('_periods')]
    for col in price_diff_cols:
        combined_df[col] = np.log(np.abs(combined_df[col]) + 1e-8) * np.sign(combined_df[col])
    
    #------------Convert timestamp to datetime and group
    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
    grouped = combined_df.groupby('timestamp')
    
    return grouped, combined_df
#-----------------------------------------------------------------------------
def load_and_group_test_data():
    data_dir = Path("../data/processed")
    
    #------------Load all processed data files
    all_data = []
    for file in data_dir.glob("option_data_with_future_prices_7_9*.csv"):
        df = pd.read_csv(file)
        all_data.append(df)
    
    #------------Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    #------------Apply log transformation to price_diff columns
    price_diff_cols = [col for col in combined_df.columns if col.startswith('price_diff_') and col.endswith('_periods')]
    for col in price_diff_cols:
        combined_df[col] = np.log(np.abs(combined_df[col]) + 1e-8) * np.sign(combined_df[col])
    
    #------------Convert timestamp to datetime and group
    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
    grouped = combined_df.groupby('timestamp')
    
    return grouped, combined_df
#-----------------------------------------------------------------------------
def prepare_lstm_data(period, type = 'train'):
    """Prepare data for LSTM training using full option matrix at each timestamp"""
    grouped, df = load_and_group_train_data() if type == 'train' else load_and_group_test_data()
    
    #------------Apply standardization to all data
    apply_standardization(df)
    
    #------------Clean data: remove NaN/inf values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df['moneyness'] = (df['strike'] - df['price']) / df['price']
    grouped = df.groupby('timestamp')
    
    #------------Create option matrices for each timestamp
    lstm_data = []
    max_options = 20  # Fixed matrix size for all timestamps
    
    for timestamp, group in grouped:
        if len(group) < 5:
            continue
            
        group_sorted = group.sort_values('strike')
        
        #--------Create option matrix (each row = one option, columns = features)
        feature_cols = ['standardized_price', 'gamma', 'theta', 'vega', 'rho', 'implied_volatility']#. 'moneyness','delta']
        option_matrix = group_sorted[feature_cols].values
        
        #--------Clip extreme values and pad/truncate to fixed size
        option_matrix = np.clip(option_matrix, -10, 10)
        
        if len(option_matrix) > max_options:
            option_matrix = option_matrix[:max_options]  # Truncate
        elif len(option_matrix) < max_options:
            # Pad with zeros
            padding = np.zeros((max_options - len(option_matrix), len(feature_cols)))
            option_matrix = np.vstack([option_matrix, padding])
        
        price_diffs = group_sorted[f'price_diff_{period}_periods'].values
        target = np.tanh(np.mean(price_diffs) / np.std(price_diffs + 1e-8))
        
        if not (np.isnan(target) or np.isinf(target)):
            lstm_data.append((option_matrix, target, timestamp))
    
    return lstm_data

#-----------------------------------------------------------------------------
def build_flat_lstm_model(input_shape):
    """Build improved LSTM model with attention mechanism"""
    
    model = Sequential([
        Reshape((input_shape[0], input_shape[1] * input_shape[2]), input_shape=input_shape),
        LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=l2(0.001)),
        LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
        BatchNormalization(),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.4),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.0005, clipnorm=1.0), 
                  loss='huber', metrics=['mae', 'mse'])
    return model
#-------------------------------------------------------------------------------
def build_matrix_lstm_model(input_shape):
    """Build LSTM model that processes option matrices directly"""

    
    model = Sequential([
        #--------Process each option matrix with TimeDistributed Dense layers
        TimeDistributed(Dense(64, activation='relu'), input_shape=input_shape),
        TimeDistributed(Dense(32, activation='relu')),
        TimeDistributed(Flatten()),
        
        #--------LSTM layers to process temporal sequences
        LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=l2(0.001)),
        LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
        BatchNormalization(),
        
        #--------Final prediction layers
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.4),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.0005, clipnorm=1.0), 
                  loss='huber', metrics=['mae', 'mse'])
    return model

#-----------------------------------------------------------------------------
def train_lstm_model(period: 3 | 6 | 12 | 15 | 30 ):
    """Train LSTM model on option data"""
    print("Preparing LSTM data...")
    lstm_data = prepare_lstm_data(period)
    
    if len(lstm_data) < 10:
        print("Not enough data for LSTM training")
        return None
    
    #------------Prepare sequences for LSTM
    X, y = [], []
    sequence_length = 5
    
    for i in range(len(lstm_data) - sequence_length):
        #--------Create sequence of option matrices
        sequence_matrices = []
        
        for j in range(sequence_length):
            option_matrix, target, _ = lstm_data[i + j]
            sequence_matrices.append(option_matrix)  # Keep as matrix
        
        X.append(np.array(sequence_matrices))
        y.append(lstm_data[i + sequence_length - 1][1])  # Predict last target
    
    X, y = np.array(X), np.array(y)
    print(f"Training data shape: X={X.shape}, y={y.shape}")
    
    #------------Robust scaling with outlier handling
    from sklearn.preprocessing import RobustScaler
    scaler_X = RobustScaler()
    original_shape = X.shape
    X_reshaped = X.reshape(-1, X.shape[-1])
    X_scaled = scaler_X.fit_transform(X_reshaped).reshape(original_shape)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=False)
    
    scaler_y = RobustScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    #------------Build and train model
    model = build_flat_lstm_model((X_train.shape[1], X_train.shape[2], X_train.shape[3]))
    
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(patience=8, factor=0.5, min_lr=1e-6, monitor='val_loss')
    ]
    
    print("Training LSTM model...")
    history = model.fit(X_train, y_train_scaled, 
                       validation_data=(X_test, y_test_scaled),
                       epochs=100, batch_size=16, verbose=1, callbacks=callbacks)
    
    #------------Evaluate model
    train_loss = model.evaluate(X_train, y_train_scaled, verbose=0)
    test_loss = model.evaluate(X_test, y_test_scaled, verbose=0)
    
    print(f"\nModel Performance:")
    print(f"Train Loss: {train_loss[0]:.4f}, Train MAE: {train_loss[1]:.4f}")
    print(f"Test Loss: {test_loss[0]:.4f}, Test MAE: {test_loss[1]:.4f}")
    ## Save model
    #model.save("trained_lstm_model.h5")
#
    ## Save scalers
    #dump(scaler_X, "scaler_X.joblib")
    #dump(scaler_y, "scaler_y.joblib")
#
    #print("Model and scalers saved.")
    return model, scaler_y, history

#-----------------------------------------------------------------------------
def analyze_grouped_data():
    """Analyze the grouped data structure"""
    grouped, df = load_and_group_train_data()
    
    print("Data Analysis:")
    print("#" + "="*60)
    print(f"Total records: {len(df)}")
    print(f"Unique timestamps: {len(grouped)}")
    print(f"Unique symbols: {len(df['symbol'].unique())}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    #------------Show available timestamps
    print("\nAvailable timestamps:")
    for i, ts in enumerate(list(grouped.groups.keys())[:5]):
        group_size = len(grouped.get_group(ts))
        calls = len(grouped.get_group(ts)[grouped.get_group(ts)['option_type'] == 'call'])
        puts = len(grouped.get_group(ts)[grouped.get_group(ts)['option_type'] == 'put'])
        print(f"    [{i}] {ts} - {group_size} options ({calls} calls, {puts} puts)")
    if len(grouped) > 5:
        print(f"    ... and {len(grouped) - 5} more timestamps")
    return len(grouped)

#-----------------------------------------------------------------------------
if __name__ == "__main__":
    #------------Analyze the data structure
    grouped_ts = analyze_grouped_data()
    
    #------------Train LSTM model
    print("\n" + "#"*70)
    print("Training LSTM model for option price prediction...")
    model, scaler, history = train_lstm_model()
    