# Streamlit Dashboard

A Streamlit application with interactive date range sliders and automatically updating charts.

## Features

- **Date Range Sliders**: Select start and end dates with smooth slider controls
- **Interactive Charts**: Price and volume charts that automatically update based on the selected date range
- **Live Metrics**: Display key statistics (start price, end price, price change, average volume)
- **Raw Data View**: Expandable section to view the underlying data

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the App

Start the Streamlit app:

```bash
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```

The app will open in your browser at `http://localhost:8501`

## Running from Docker or WSL

When the app runs inside a Docker container, Streamlit must bind to all interfaces
and Docker must publish the port to Windows:

```bash
docker run -p 8501:8501 <your-image>
```

If the container is already running without `-p 8501:8501`, stop and recreate it
with the port mapping. Inside the container, run:

```bash
cd streamlit_app
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```

Then open `http://localhost:8501` on Windows.

## Usage

1. Adjust the **Start Date** slider to select the beginning of your date range
2. Adjust the **End Date** slider to select the end of your date range
3. The charts and metrics automatically update to reflect the selected period

## Data

The app uses sample generated data for demonstration purposes. You can modify the `generate_sample_data()` function to load your own data.
