import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from lstm_model import load_and_group_test_data, apply_standardization, train_lstm_model, prepare_lstm_data
from sklearn.preprocessing import RobustScaler
import warnings, random
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
warnings.filterwarnings('ignore')
DEFAULT_PERIOD=3
TEST_SIZE = 100

def get_model_predictions(model, scaler, df, sample_data_size, period):
    """Generate actual model predictions for sample data"""
    try:
        lstm_data = prepare_lstm_data(period, 'test')
        if len(lstm_data) < 10:
            return Exception("Not enough data for prediction")
        
        #--------Prepare sequences matching training format
        X = []
        sequence_length = 5
        for i in range(min(sample_data_size, len(lstm_data) - sequence_length)):
            sequence_matrices = [lstm_data[i + j][0] for j in range(sequence_length)]
            X.append(np.array(sequence_matrices))
        
        if len(X) == 0:
            return Exception("No valid sequences found")
            
        X = np.array(X)
        
        #--------Apply same scaling as training
        scaler_X = RobustScaler()
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = scaler_X.fit_transform(X_reshaped).reshape(original_shape)
        
        predictions = model.predict(X_scaled, verbose=0)
        pred_scaled = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        
        return pred_scaled
    except Exception as e:
        print(f"Error in prediction: {e}")
        exit(1)
#-----------------------------------------------------------------------------
def create_dashboard():
    """Create dashboard to evaluate LSTM model accuracy"""
    
    #------------Load and prepare data
    print("Loading data and training model...")
    model, scaler, history = train_lstm_model(period=DEFAULT_PERIOD)
    
    if model is None:
        print("Model training failed")
        return None
    
    #------------Get evaluation data
    grouped, df = load_and_group_test_data()
    apply_standardization(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    #------------Create Dash app
    app = dash.Dash(__name__)
    
    app.layout = html.Div([
        html.H1("LSTM Option Price Prediction Dashboard", 
                style={'textAlign': 'center', 'marginBottom': 30}),
        
        #--------Training Performance Section
        html.Div([
            html.H2("Model Training Performance"),
            dcc.Graph(id='training-loss-graph'),
            dcc.Graph(id='training-metrics-graph')
        ], style={'marginBottom': 40}),
        
        #--------Prediction Accuracy Section
        html.Div([
            html.H2("Prediction vs Actual Analysis"),
            dcc.Graph(id='prediction-scatter'),
            dcc.Graph(id='residuals-plot')
        ], style={'marginBottom': 40}),
        
        #--------Time Series Analysis
        html.Div([
            html.H2("Time Series Prediction Accuracy"),
            dcc.Graph(id='time-series-plot'),
            dcc.Graph(id='error-distribution')
        ])
    ])
    
    #------------Callback for training loss
    @app.callback(
        Output('training-loss-graph', 'figure'),
        Input('training-loss-graph', 'id')
    )
    def update_training_loss(_):
        if history is None:
            return go.Figure()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=history.history['loss'],
            mode='lines',
            name='Training Loss',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            y=history.history['val_loss'],
            mode='lines',
            name='Validation Loss',
            line=dict(color='red')
        ))
        fig.update_layout(
            title='Model Training Loss Over Epochs',
            xaxis_title='Epoch',
            yaxis_title='Loss (MSE)',
            hovermode='x'
        )
        return fig
    
    #------------Callback for training metrics
    @app.callback(
        Output('training-metrics-graph', 'figure'),
        Input('training-metrics-graph', 'id')
    )
    def update_training_metrics(_):
        if history is None:
            return go.Figure()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=history.history['mae'],
            mode='lines',
            name='Training MAE',
            line=dict(color='green')
        ))
        fig.add_trace(go.Scatter(
            y=history.history['val_mae'],
            mode='lines',
            name='Validation MAE',
            line=dict(color='orange')
        ))
        fig.update_layout(
            title='Model Training MAE Over Epochs',
            xaxis_title='Epoch',
            yaxis_title='Mean Absolute Error',
            hovermode='x'
        )
        return fig
    
    #------------Callback for prediction scatter
    @app.callback(
        Output('prediction-scatter', 'figure'),
        Input('prediction-scatter', 'id')
    )
    def update_prediction_scatter(_):
        #--------Generate sample predictions for visualization
        sample_data = df.groupby('timestamp').apply(
            lambda x: x[f'price_diff_{DEFAULT_PERIOD}_periods'].mean()
        ).dropna().values[:TEST_SIZE]
        
        #--------Use actual model predictions
        predictions = get_model_predictions(model, scaler, df, len(sample_data), DEFAULT_PERIOD)
        
        #--------Ensure same length
        #min_len = min(len(sample_data), len(predictions))
        #sample_data = sample_data[:min_len]
        #predictions = predictions[:min_len]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sample_data,
            y=predictions,
            mode='markers',
            name='Predictions vs Actual',
            marker=dict(color='blue', opacity=0.6)
        ))
        
        X = sample_data.reshape(-1, 1)
        y = predictions
        lr = LinearRegression().fit(X, y)
        y_fit = lr.predict(X)

        line_x = np.array([sample_data.min(), sample_data.max()]).reshape(-1, 1)
        line_y = lr.predict(line_x)
        fig.add_trace(go.Scatter(
            x=line_x.flatten(),
            y=line_y,
            mode='lines',
            name='Linear Fit',
            line=dict(color='green', width=2)
        ))
        
        #--------Add perfect prediction line
        min_val, max_val = min(sample_data), max(sample_data)
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='Predicted vs Actual Price Differences',
            xaxis_title='Actual Price Diff',
            yaxis_title='Predicted Price Diff',
            hovermode='closest'
        )
        return fig
    
    #------------Callback for residuals plot
    @app.callback(
        Output('residuals-plot', 'figure'),
        Input('residuals-plot', 'id')
    )
    def update_residuals(_):
        #--------Generate sample residuals
        ts_data = df.groupby('timestamp').apply(
            lambda x: x[f'price_diff_{DEFAULT_PERIOD}_periods'].mean()
        ).dropna()
        
        sample_data = ts_data.values[:min(200, len(ts_data))]
        predictions = get_model_predictions(model, scaler, df, len(sample_data), DEFAULT_PERIOD)
        
        residuals = np.array(sample_data) - np.array(predictions[:len(sample_data)])
        
        squared_residuals = residuals
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=predictions[:len(sample_data)],
            y=squared_residuals,
            mode='markers',
            name='Residuals',
            marker=dict(color='purple', opacity=0.6, size=6)
        ))
        
        fig.update_layout(
            title='Residuals vs Predicted Values',
            xaxis_title='Predicted Values',
            yaxis_title='Residuals',
            hovermode='closest'
        )
        return fig
    
    #------------Callback for time series plot
    @app.callback(
        Output('time-series-plot', 'figure'),
        Input('time-series-plot', 'id')
    )
    def update_time_series(_):
        #--------Get time series data
        ts_data = df.groupby('timestamp').agg({
            f'price_diff_{DEFAULT_PERIOD}_periods': 'mean'
        }).reset_index()
        ts_data = ts_data.dropna()[:100]  # Show first 100 timestamps
        
        # Use actual model predictions
        actual_values = ts_data[f'price_diff_{DEFAULT_PERIOD}_periods'].values
        predictions = get_model_predictions(model, scaler, df, len(actual_values), DEFAULT_PERIOD)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ts_data['timestamp'],
            y=ts_data[f'price_diff_{DEFAULT_PERIOD}_periods'],
            mode='lines+markers',
            name='Actual',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=ts_data['timestamp'],
            y=predictions,
            mode='lines+markers',
            name='Predicted',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='Time Series: Actual vs Predicted Price Differences',
            xaxis_title='Timestamp',
            yaxis_title='Price Difference',
            hovermode='x'
        )
        return fig
    
    #------------Callback for error distribution
    @app.callback(
        Output('error-distribution', 'figure'),
        Input('error-distribution', 'id')
    )
    def update_error_distribution(_):
        #--------Generate error distribution
        sample_data = df.groupby('timestamp').apply(
            lambda x: x[f'price_diff_{DEFAULT_PERIOD}_periods'].mean()
        ).dropna().values[:TEST_SIZE]
        
        if len(sample_data) == 0:
            return go.Figure().update_layout(title='No data available for error distribution')
        
        predictions = get_model_predictions(model, scaler, df, len(sample_data), DEFAULT_PERIOD)
        
        # Ensure arrays have same length
        min_len = min(len(sample_data), len(predictions))
        errors = sample_data[:min_len] - predictions[:min_len]
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=errors,
            nbinsx=30,
            name='Prediction Errors',
            marker_color='lightblue',
            opacity=0.7
        ))
        
        fig.update_layout(
            title='Distribution of Prediction Errors',
            xaxis_title='Prediction Error',
            yaxis_title='Frequency',
            showlegend=False
        )
        return fig
    
    return app

#-----------------------------------------------------------------------------
if __name__ == "__main__":
    app = create_dashboard()
    if app:
        print("Starting dashboard server...")
        app.run(debug=True, port=8050)
    else:
        print("Failed to create dashboard")