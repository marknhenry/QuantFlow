import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Page configuration
st.set_page_config(page_title="QuantFlow Dashboard", layout="wide")

st.title("📊 QuantFlow Dashboard")

# Generate sample data
@st.cache_data
def generate_sample_data():
    """Generate sample time series data for demonstration"""
    start = datetime(2023, 1, 1)
    end = datetime(2024, 12, 31)
    dates = pd.date_range(start, end, freq='D')
    
    np.random.seed(42)
    data = pd.DataFrame({
        'date': dates,
        'price': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        'volume': np.random.randint(1000000, 5000000, len(dates))
    })
    return data

# Load data
data = generate_sample_data()
min_date = data['date'].min().date()
max_date = data['date'].max().date()

# Create two columns for sliders
col1, col2 = st.columns(2)

with col1:
    start_date = st.slider(
        "📅 Start Date",
        min_value=min_date,
        max_value=max_date,
        value=min_date,
        step=timedelta(days=1),
        key="start_date"
    )

with col2:
    end_date = st.slider(
        "📅 End Date",
        min_value=min_date,
        max_value=max_date,
        value=max_date,
        step=timedelta(days=1),
        key="end_date"
    )

# Validate date range
if start_date >= end_date:
    st.error("❌ Start date must be before end date!")
    st.stop()

# Filter data based on date range
filtered_data = data[(data['date'].dt.date >= start_date) & (data['date'].dt.date <= end_date)]

# Display metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📈 Start Price", f"${filtered_data['price'].iloc[0]:.2f}")

with col2:
    st.metric("🏁 End Price", f"${filtered_data['price'].iloc[-1]:.2f}")

with col3:
    price_change = filtered_data['price'].iloc[-1] - filtered_data['price'].iloc[0]
    st.metric("📊 Change", f"${price_change:.2f}", delta=f"{(price_change/filtered_data['price'].iloc[0]*100):.2f}%")

with col4:
    avg_volume = filtered_data['volume'].mean()
    st.metric("💹 Avg Volume", f"{avg_volume/1e6:.2f}M")

# Create interactive chart
st.subheader("Price Chart")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=filtered_data['date'],
    y=filtered_data['price'],
    mode='lines',
    name='Price',
    line=dict(color='#1f77b4', width=2),
    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Price: $%{y:.2f}<extra></extra>'
))

fig.update_layout(
    title=f"Price History ({start_date} to {end_date})",
    xaxis_title="Date",
    yaxis_title="Price ($)",
    hovermode='x unified',
    template="plotly_white",
    height=500,
    margin=dict(l=0, r=0, t=40, b=0)
)

st.plotly_chart(fig, use_container_width=True)

# Volume chart
st.subheader("Trading Volume")

fig2 = go.Figure()

fig2.add_trace(go.Bar(
    x=filtered_data['date'],
    y=filtered_data['volume'],
    name='Volume',
    marker=dict(color='#ff7f0e'),
    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Volume: %{y:,.0f}<extra></extra>'
))

fig2.update_layout(
    xaxis_title="Date",
    yaxis_title="Volume",
    hovermode='x unified',
    template="plotly_white",
    height=350,
    margin=dict(l=0, r=0, t=40, b=0),
    showlegend=False
)

st.plotly_chart(fig2, use_container_width=True)

# Display raw data
with st.expander("📋 View Raw Data"):
    st.dataframe(filtered_data.reset_index(drop=True), use_container_width=True)
