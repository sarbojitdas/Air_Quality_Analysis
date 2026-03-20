import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

st.set_page_config(page_title="Air Quality Dashboard", layout="wide")

# -----------------------------
# LOAD DATA (CACHED)
# -----------------------------
@st.cache_data
def load_data():
    station_day = pd.read_csv("data/station_day.csv")
    stations = pd.read_csv("data/stations.csv")

    # Clean columns
    stations.columns = stations.columns.str.strip().str.lower()
    station_day.columns = station_day.columns.str.strip()

    df = pd.merge(
        station_day,
        stations[['stationid', 'stationname', 'city']],
        left_on='StationId',
        right_on='stationid',
        how='left'
    )

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.fillna(method='ffill')

    return df

df = load_data()

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("🔍 Filters")

cities = sorted(df['city'].dropna().unique())
selected_city = st.sidebar.selectbox("Select City", cities)

df_city = df[df['city'] == selected_city]

stations_list = sorted(df_city['stationname'].dropna().unique())
selected_station = st.sidebar.selectbox("Select Station", stations_list)

df_station = df_city[df_city['stationname'] == selected_station]

# -----------------------------
# HEADER
# -----------------------------
st.title("🌍 Air Quality Dashboard")
st.markdown(f"### 📍 {selected_city} | 🏭 {selected_station}")

# -----------------------------
# AQI TREND
# -----------------------------
st.subheader("📈 AQI Trend")

fig = px.line(df_station, x='Date', y='AQI')
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# TOP STATIONS
# -----------------------------
st.subheader("🏭 Top Polluted Stations")

top_stations = (
    df.groupby(['stationname', 'city'])['AQI']
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig2 = px.bar(top_stations, x='stationname', y='AQI', color='city')
st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# MAP (SIMPLIFIED)
# -----------------------------
st.subheader("🗺️ AQI Map")

city_coords = {
    "Kolkata": [22.5726, 88.3639],
    "Delhi": [28.7041, 77.1025],
    "Mumbai": [19.0760, 72.8777],
    "Bengaluru": [12.9716, 77.5946],
    "Chennai": [13.0827, 80.2707],
    "Hyderabad": [17.3850, 78.4867]
}

map_df = df.copy()
map_df['lat'] = map_df['city'].map(lambda x: city_coords.get(x, [None, None])[0])
map_df['lon'] = map_df['city'].map(lambda x: city_coords.get(x, [None, None])[1])

map_df = map_df.dropna(subset=['lat', 'lon'])

map_df = map_df.groupby('city').agg({
    'AQI': 'mean',
    'lat': 'first',
    'lon': 'first'
}).reset_index()

fig3 = px.scatter_map(
    map_df,
    lat="lat",
    lon="lon",
    color="AQI",
    size="AQI",
    hover_name="city",
    zoom=4
)

st.plotly_chart(fig3, use_container_width=True)

# -----------------------------
# 🔮 PREDICTION
# -----------------------------
st.subheader("🔮 Future AQI Prediction")

col1, col2, col3 = st.columns(3)

hour = col1.slider("Hour", 0, 23, 12)
day = col2.slider("Day", 1, 31, 15)
month = col3.slider("Month", 1, 12, 6)

input_data = np.array([[hour, day, month]])

prediction = model.predict(input_data)[0]

st.metric("Predicted AQI", round(prediction, 2))