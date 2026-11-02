import os
import requests
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Weather Forecasting App", page_icon="🌦")


API_KEY = os.getenv("OPENWEATHER_API_KEY")
if not API_KEY:
    try:
        API_KEY = st.secrets["OPENWEATHER_API_KEY"]
    except Exception:
        API_KEY = None


@st.cache_resource
def load_models():
    return joblib.load("rain_model.pkl"), joblib.load("temp_model.pkl")


rain_model, temp_model = load_models()

RAIN_FEATURES = ["MinTemp", "MaxTemp", "Humidity9am",
                 "Humidity3pm", "Pressure9am", "Pressure3pm"]
TEMP_FEATURES = ["MinTemp", "Humidity9am", "Humidity3pm",
                 "Pressure9am", "Pressure3pm", "Temp9am"]


def get_weather(city):
    r = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={"q": city, "appid": API_KEY, "units": "metric"},
        timeout=10,
    )
    r.raise_for_status()
    d = r.json()
    return {
        "temp": d["main"]["temp"],
        "temp_min": d["main"]["temp_min"],
        "temp_max": d["main"]["temp_max"],
        "humidity": d["main"]["humidity"],
        "pressure": d["main"]["pressure"],
    }


st.title("🌦 Weather Forecasting App")
st.write("Fetches live weather and predicts tomorrow's rain and temperature.")

if not API_KEY:
    st.error("API key not found. Add OPENWEATHER_API_KEY in the app's Secrets.")
    st.stop()

city = st.text_input("City", "Meerut")

if st.button("Predict Tomorrow"):
    try:
        w = get_weather(city)
    except requests.HTTPError as e:
        code = e.response.status_code
        if code == 404:
            st.error("City not found. Check the spelling.")
        elif code == 401:
            st.error("Invalid API key.")
        else:
            st.error(f"API error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Could not fetch weather: {e}")
        st.stop()

    row = {
        "MinTemp": w["temp_min"],
        "MaxTemp": w["temp_max"],
        "Humidity9am": w["humidity"],
        "Humidity3pm": w["humidity"],
        "Pressure9am": w["pressure"],
        "Pressure3pm": w["pressure"],
        "Temp9am": w["temp"],
    }
    data = pd.DataFrame([row])

    rain_pred = rain_model.predict(data[RAIN_FEATURES])[0]
    temp_pred = temp_model.predict(data[TEMP_FEATURES])[0]

    st.subheader(f"Current weather in {city}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Temperature", f"{w['temp']:.1f} °C")
    c2.metric("Humidity", f"{w['humidity']}%")
    c3.metric("Pressure", f"{w['pressure']} hPa")

    st.subheader("Prediction")
    if rain_pred == 1:
        st.success("🌧 Rain Expected Tomorrow")
    else:
        st.success("☀ No Rain Expected Tomorrow")
    st.info(f"🌡 Predicted Max Temperature: {temp_pred:.2f} °C")
