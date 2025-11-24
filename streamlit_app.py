import streamlit as st
import requests
import joblib
import numpy as np

# --------------------------
# PAGE CONFIG (TITLE + ICON)
# --------------------------
st.set_page_config(
    page_title="AI Air Quality Prediction",
    page_icon="🌍",
    layout="centered"
)

# --------------------------
# BACKGROUND IMAGE USING CSS
# --------------------------
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1502134249126-9f3755a50d78");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# --------------------------
# LOAD TRAINED MODEL
# --------------------------
pipe = joblib.load("aqi_model.pkl")

# --------------------------
# UI TITLE
# --------------------------
st.markdown("<h1 style='text-align:center; color:white;'>🌫️ AI-Powered Air Quality Prediction</h1>", unsafe_allow_html=True)

# --------------------------
# SIDEBAR FOR USER INPUT
# --------------------------
st.sidebar.header("🔑 OpenWeather API Setup")
API_KEY = st.sidebar.text_input("Enter your OpenWeather API Key", type="password")
LAT = st.sidebar.text_input("Latitude", "13.9299")
LON = st.sidebar.text_input("Longitude", "75.5681")

# --------------------------
# FUNCTION TO FETCH LIVE DATA
# --------------------------
def fetch_openweather(api, lat, lon):
    try:
        url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api}"
        res = requests.get(url)
        res.raise_for_status()
        data = res.json()

        comp = data["list"][0]["components"]
        aqi = data["list"][0]["main"]["aqi"]
        return comp, aqi, url

    except Exception as e:
        return None, None, str(e)

# --------------------------
# MAIN BUTTON
# --------------------------
if st.button("🔄 Fetch Live Pollution Data & Predict AQI"):
    if not API_KEY:
        st.error("❌ Please enter your API key")
    else:
        comp, ow_aqi, info = fetch_openweather(API_KEY, LAT, LON)

        if comp is None:
            st.error("API Error: " + info)
        else:
            st.success("Live Data Fetched Successfully!")
            st.json(comp)

            # Extract the 8 required features
            co    = comp["co"]
            no    = comp["no"]
            no2   = comp["no2"]
            o3    = comp["o3"]
            so2   = comp["so2"]
            pm25  = comp["pm2_5"]
            pm10  = comp["pm10"]
            nh3   = comp["nh3"]

            # Correct 8-feature input to model
            X = np.array([[co, no, no2, o3, so2, pm25, pm10, nh3]])

            # Model prediction
            pred = pipe.predict(X)[0]

            st.write("### 🤖 Model Predicted AQI Value")
            st.info(f"**{pred:.2f}**")

            st.write("### 📊 OpenWeather AQI Scale Value")
            st.warning(f"**{ow_aqi}** (1 = Good, 5 = Very Poor)")

            st.write("### 🔍 API URL Used")
            st.code(info)

# --------------------------
# FOOTER
# --------------------------
st.markdown("<br><br><center><p style='color:white;'>Developed by AI 🤖</p></center>", unsafe_allow_html=True)
