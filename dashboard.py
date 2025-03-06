import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from agri_analytics import AgriAnalytics
from anomaly_detector import AnomalyDetector, IsolationForestStrategy

# Initialize the analytics engine
analytics = AgriAnalytics()

# Page configuration
st.set_page_config(page_title="AgriSeeds Analytics Dashboard", layout="wide")

# Title and description
st.title("AgriSeeds Analytics Dashboard")
st.markdown("Real-time agricultural market analytics and insights")

# Sidebar configuration
st.sidebar.header("Settings")

# Data refresh rate
refresh_interval = st.sidebar.slider(
    "Data Refresh Interval (seconds)",
    min_value=30,
    max_value=300,
    value=60
)

# Alert threshold configuration
price_change_threshold = st.sidebar.slider(
    "Price Change Alert Threshold (%)",
    min_value=1.0,
    max_value=10.0,
    value=5.0
)

# Load and process data
@st.cache_data(ttl=refresh_interval)
def load_data():
    data = analytics.load_futures_data('futures_prices.csv')
    return data

data = load_data()

# Create main dashboard layout
col1, col2 = st.columns(2)

# Price Trends Chart
with col1:
    st.subheader("Price Trends")
    fig = px.line(data, x='Date', y='Close', title='Futures Price Trends')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# Anomaly Detection
with col2:
    st.subheader("Anomaly Detection")
    anomaly_data, stats = analytics.detect_anomalies()
    
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(
        go.Scatter(x=data['Date'], y=data['Close'], name="Price", line=dict(color='blue'))
    )
    
    # Add anomalies as scatter points
    anomaly_points = data[data['is_anomaly'] == -1]
    fig.add_trace(
        go.Scatter(
            x=anomaly_points['Date'],
            y=anomaly_points['Close'],
            mode='markers',
            name='Anomalies',
            marker=dict(color='red', size=10)
        )
    )
    
    fig.update_layout(
        title='Price Anomalies Detection',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# Statistics and Metrics
col3, col4, col5 = st.columns(3)

with col3:
    st.metric(
        label="Current Price",
        value=f"${data['Close'].iloc[-1]:.2f}",
        delta=f"{(data['Close'].iloc[-1] - data['Close'].iloc[-2]):.2f}"
    )

with col4:
    st.metric(
        label="Daily Change",
        value=f"{((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100):.2f}%"
    )

with col5:
    st.metric(
        label="Anomalies Detected",
        value=len(anomaly_data)
    )

# Alert System
st.subheader("Recent Alerts")

def check_alerts(data, threshold):
    alerts = []
    daily_change = ((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100)
    
    if abs(daily_change) >= threshold:
        alerts.append(f"⚠️ Significant price change detected: {daily_change:.2f}%")
    
    if len(data[data['is_anomaly'] == -1]) > 0:
        alerts.append(f"🔍 Anomaly detected in recent price movements")
    
    return alerts

alerts = check_alerts(data, price_change_threshold)

if alerts:
    for alert in alerts:
        st.warning(alert)
else:
    st.info("No active alerts at this time")

# Auto-refresh the dashboard
st.empty()
st.markdown(f"*Dashboard auto-refreshes every {refresh_interval} seconds*")