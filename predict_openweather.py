import requests
import joblib
import numpy as np

API_KEY = "7ad15e158a0506009eb5cff33ec096a4"     # <-- FIX THIS

LAT = 13.9299
LON = 75.5681

pipe = joblib.load("aqi_model.pkl")  # Use your actual model file # <-- FIX THIS

def get_components(lat, lon, key):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={key}"
    print("\n🔍 URL:", url)
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()
    return data["list"][0]["components"], data["list"][0]["main"]["aqi"]

comp, ow_aqi = get_components(LAT, LON, API_KEY)

print("OpenWeather Components:", comp)
print("OpenWeather AQI:", ow_aqi)

input_data = np.array([[comp["co"], comp["no"], comp["no2"], comp["o3"], comp["so2"],
                        comp["pm2_5"], comp["pm10"], comp["nh3"]]])

predicted_aqi = pipe.predict(input_data)[0]
print("\nPredicted AQI:", predicted_aqi)
