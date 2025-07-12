#!/usr/bin/env python3
"""
Real-time SPY Option Divergence Dashboard
Displays live divergence metrics and alerts
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import glob
from datetime import datetime, timedelta
import time

#----------------------------Page configuration
st.set_page_config(
    page_title="SPY Option Divergence Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data(ttl=30)  # Cache for 30 seconds
def load_latest_data():
    """Load the most recent option and divergence data"""
    
    #------------ Load option data
    option_files = glob.glob('data/option_data_*.csv')
    if option_files:
        latest_option_file = max(option_files, key=lambda x: datetime.fromtimestamp(os.path.getmtime(x)))
        option_df = pd.read_csv(latest_option_file)
        option_df['timestamp'] = pd.to_datetime(option_df['timestamp'])
    else:
        option_df = pd.DataFrame()
    
    #------------ Load divergence alerts
    alert_files = glob.glob('data/divergence_alerts_*.json')
    alerts = []
    
    for file in alert_files:
        try:
            with open(file, 'r') as f:
                for line in f:
                    if line.strip():
                        alerts.append(json.loads(line.strip()))
        except Exception as e:
            st.error(f"Error loading alerts from {file}: {e}")
    
    alerts_df = pd.DataFrame(alerts) if alerts else pd.DataFrame()
    if not alerts_df.empty:
        alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'])
    
    #------------ Load divergence analysis data
    divergence_files = glob.glob('data/divergence_data_*.csv') + glob.glob('data/atm_divergence_analysis.csv')
    if divergence_files:
        latest_div_file = max(divergence_files, key=lambda x: datetime.fromtimestamp(os.path.getmtime(x)))
        divergence_df = pd.read_csv(latest_div_file)
        if 'timestamp' in divergence_df.columns:
            divergence_df['timestamp'] = pd.to_datetime(divergence_df['timestamp'])
    else:
        divergence_df = pd.DataFrame()
    
    return option_df, alerts_df, divergence_df

def create_price_chart(option_df):
    """Create SPY price evolution chart"""
    if option_df.empty:
        return go.Figure()
    
    spy_prices = option_df[['timestamp', 'price']].drop_duplicates().sort_values('timestamp')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=spy_prices['timestamp'],
        y=spy_prices['price'],
        mode='lines',
        name='SPY Price',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title="SPY Price Evolution",
        xaxis_title="Time",
        yaxis_title="Price ($)",
        height=400
    )
    
    return fig

def create_divergence_chart(divergence_df, alerts_df):
    """Create divergence metrics chart"""
    if divergence_df.empty:
        return go.Figure()
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Price Divergence', 'IV Divergence', 'Put-Call Parity Divergence'),
        vertical_spacing=0.08
    )
    
    #------------ Price divergence
    fig.add_trace(
        go.Scatter(x=divergence_df['timestamp'], y=divergence_df['price_divergence'],
                  mode='lines', name='Price Div', line=dict(color='red')),
        row=1, col=1
    )
    
    #------------ IV divergence
    fig.add_trace(
        go.Scatter(x=divergence_df['timestamp'], y=divergence_df['iv_divergence'],
                  mode='lines', name='IV Div', line=dict(color='green')),
        row=2, col=1
    )
    
    #------------ Parity divergence
    fig.add_trace(
        go.Scatter(x=divergence_df['timestamp'], y=divergence_df['parity_divergence'],
                  mode='lines', name='Parity Div', line=dict(color='purple')),
        row=3, col=1
    )
    
    #------------ Add alert markers if available
    if not alerts_df.empty:
        for _, alert in alerts_df.iterrows():
            color = 'green' if alert['signal'] == 1 else 'red'
            for row in [1, 2, 3]:
                fig.add_vline(
                    x=alert['timestamp'],
                    line=dict(color=color, width=2, dash='dash'),
                    row=row, col=1
                )
    
    #------------ Add zero lines
    for row in [1, 2, 3]:
        fig.add_hline(y=0, line=dict(color='gray', width=1, dash='dot'), row=row, col=1)
    
    fig.update_layout(height=800, showlegend=False)
    return fig

def create_signal_distribution(alerts_df):
    """Create signal distribution chart"""
    if alerts_df.empty:
        return go.Figure()
    
    signal_counts = alerts_df['signal_name'].value_counts()
    
    fig = go.Figure(data=[
        go.Bar(x=signal_counts.index, y=signal_counts.values,
               marker_color=['red' if x == 'SELL' else 'green' if x == 'BUY' else 'gray' 
                           for x in signal_counts.index])
    ])
    
    fig.update_layout(
        title="Signal Distribution",
        xaxis_title="Signal Type",
        yaxis_title="Count",
        height=300
    )
    
    return fig

def create_divergence_heatmap(divergence_df):
    """Create divergence correlation heatmap"""
    if divergence_df.empty:
        return go.Figure()
    
    #------------ Create correlation matrix of divergence metrics
    metrics = ['price_divergence', 'iv_divergence', 'parity_divergence']
    corr_matrix = divergence_df[metrics].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0
    ))
    
    fig.update_layout(
        title="Divergence Metrics Correlation",
        height=400
    )
    
    return fig

#----------------------------Main dashboard
def main():
    st.title("SPY Option Divergence Monitor")
    st.markdown("Real-time monitoring of SPY option divergence patterns")
    
    #------------------------Sidebar controls
    st.sidebar.header("Controls")
    auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=True)
    refresh_button = st.sidebar.button("Refresh Now")
    
    if auto_refresh:
        #--------------------Auto-refresh every 30 seconds
        time.sleep(1)
        st.rerun()
    
    #------------------------Load data
    with st.spinner("Loading latest data..."):
        option_df, alerts_df, divergence_df = load_latest_data()
    
    #------------------------Status indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Option Records", len(option_df))
    
    with col2:
        st.metric("Total Alerts", len(alerts_df))
    
    with col3:
        if not alerts_df.empty:
            recent_alerts = alerts_df[alerts_df['timestamp'] >= alerts_df['timestamp'].max() - timedelta(hours=1)]
            st.metric("Recent Alerts (1h)", len(recent_alerts))
        else:
            st.metric("Recent Alerts (1h)", 0)
    
    with col4:
        if not option_df.empty:
            latest_spy = option_df['price'].iloc[-1]
            st.metric("Latest SPY Price", f"${latest_spy:.2f}")
        else:
            st.metric("Latest SPY Price", "N/A")
    
    #------------ Main charts
    st.header("Price Evolution")
    if not option_df.empty:
        price_chart = create_price_chart(option_df)
        st.plotly_chart(price_chart, use_container_width=True)
    else:
        st.info("No option data available")
    
    #------------ Divergence analysis
    st.header("Divergence Analysis")
    if not divergence_df.empty:
        divergence_chart = create_divergence_chart(divergence_df, alerts_df)
        st.plotly_chart(divergence_chart, use_container_width=True)
    else:
        st.info("No divergence data available. Run the analysis script to generate divergence metrics.")
    
    #------------------------Two column layout for additional charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Signal Distribution")
        if not alerts_df.empty:
            signal_chart = create_signal_distribution(alerts_df)
            st.plotly_chart(signal_chart, use_container_width=True)
        else:
            st.info("No alerts generated yet")
    
    with col2:
        st.subheader("Divergence Correlation")
        if not divergence_df.empty:
            heatmap_chart = create_divergence_heatmap(divergence_df)
            st.plotly_chart(heatmap_chart, use_container_width=True)
        else:
            st.info("No divergence data for correlation analysis")
    
    #------------------------Recent alerts table
    st.header("Recent Alerts")
    if not alerts_df.empty:
        recent_alerts = alerts_df.tail(10).sort_values('timestamp', ascending=False)
        st.dataframe(
            recent_alerts[['timestamp', 'signal_name', 'spy_price', 'price_divergence', 'iv_divergence']],
            use_container_width=True
        )
    else:
        st.info("No alerts to display")
    
    #------------ Statistical Summary
    with st.expander("Statistical Analysis"):
        if not divergence_df.empty:
            st.write("**Divergence Statistics:**")
            for metric in ['price_divergence', 'iv_divergence', 'parity_divergence']:
                if metric in divergence_df.columns:
                    mean_val = divergence_df[metric].mean()
                    std_val = divergence_df[metric].std()
                    st.write(f"- {metric}: μ={mean_val:.4f}, σ={std_val:.4f}")
        
        if not alerts_df.empty:
            st.write("**Signal Performance:**")
            signal_summary = alerts_df['signal_name'].value_counts()
            total_signals = len(alerts_df)
            for signal, count in signal_summary.items():
                pct = (count/total_signals)*100
                st.write(f"- {signal}: {count} ({pct:.1f}%)")
    
    #------------------------Footer
    st.markdown("---")
    st.markdown("*Dashboard updates automatically every 30 seconds when auto-refresh is enabled*")

if __name__ == "__main__":
    import os
    main()