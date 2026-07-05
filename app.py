import streamlit as st
import pandas as pd
import joblib
import numpy as np


st.title("🌧️ Прогнозування ймовірності дощу в Австралії")
st.subheader("📊 Параметри")


@st.cache_resource
def load_model_assets():
    try:
     
        assets = joblib.load('aussie_rain.joblib')
        return assets
    except Exception as e:
        st.error(f"Помилка завантаження файлу моделі: {e}")
        return None

assets = load_model_assets()


LOCATIONS = ['Sydney', 'Melbourne', 'Brisbane', 'Adelaide', 'Perth', 'Hobart', 'Canberra', 'Darwin', 'Other']
DIRECTIONS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'NNE', 'ENE', 'ESE', 'SSE', 'SSW', 'WSW', 'WNW', 'NNW']

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🌍 Локація та Загальне")
    location = st.selectbox("Регіон (Location)", LOCATIONS)
    rain_today = st.selectbox("Чи йшов дощ сьогодні? (RainToday)", ["No", "Yes"])
    rainfall = st.number_input("Кількість опадів сьогодні, мм (Rainfall)", min_value=0.0, value=0.0)
    evaporation = st.number_input("Випаровування, мм (Evaporation)", min_value=0.0, value=5.0)
    sunshine = st.number_input("Кількість сонячних годин (Sunshine)", min_value=0.0, max_value=16.0, value=8.0)

with col2:
    st.markdown("### 🌡️ Температура та Вологість")
    min_temp = st.slider("Мінімальна температура (°C)", -10.0, 45.0, 12.0)
    max_temp = st.slider("Максимальна температура (°C)", -10.0, 50.0, 23.0)
    temp_9am = st.slider("Температура о 9:00 (°C)", -10.0, 45.0, 17.0)
    temp_3pm = st.slider("Температура о 15:00 (°C)", -10.0, 45.0, 21.0)
    humidity_9am = st.slider("Вологість о 9:00 (%)", 0, 100, 60)
    humidity_3pm = st.slider("Вологість о 15:00 (%)", 0, 100, 50)

with col3:
    st.markdown("### 💨 Вітер, Тиск та Хмари")
    wind_gust_dir = st.selectbox("Напрямок сильного пориву вітру", DIRECTIONS)
    wind_gust_speed = st.number_input("Швидкість пориву вітру (км/год)", min_value=0.0, value=30.0)
    wind_dir_9am = st.selectbox("Напрямок вітру о 9:00", DIRECTIONS)
    wind_dir_3pm = st.selectbox("Напрямок вітру о 15:00", DIRECTIONS)
    wind_speed_9am = st.number_input("Швидкість вітру о 9:00 (км/год)", min_value=0.0, value=10.0)
    wind_speed_3pm = st.number_input("Швидкість вітру о 15:00 (км/год)", min_value=0.0, value=15.0)
    pressure_9am = st.number_input("Тиск о 9:00 (гПа)", min_value=900.0, value=1015.0)
    pressure_3pm = st.number_input("Тиск о 15:00 (гПа)", min_value=900.0, value=1012.0)
    cloud_9am = st.slider("Хмарність о 9:00 (бали: 0-8)", 0, 8, 4)
    cloud_3pm = st.slider("Хмарність о 15:00 (бали: 0-8)", 0, 8, 4)

st.write("---")

if st.button("🔮 Зробити прогноз", type="primary", use_container_width=True):
    if assets is None:
        st.error("Модель не завантажена.")
    else:
        try:
           
            model = assets['model']
            numeric_cols = assets['numeric_cols']
            encoded_cols = assets['encoded_cols']
            
            
            user_features = {
                'MinTemp': min_temp, 'MaxTemp': max_temp, 'Rainfall': rainfall, 'Evaporation': evaporation,
                'Sunshine': sunshine, 'WindGustSpeed': wind_gust_speed, 'WindSpeed9am': wind_speed_9am,
                'WindSpeed3pm': wind_speed_3pm, 'Humidity9am': humidity_9am, 'Humidity3pm': humidity_3pm,
                'Pressure9am': pressure_9am, 'Pressure3pm': pressure_3pm, 'Cloud9am': cloud_9am,
                'Cloud3pm': cloud_3pm, 'Temp9am': temp_9am, 'Temp3pm': temp_3pm
            }
            
            
            raw_numeric_values = [user_features[col] for col in numeric_cols]
            
            
            numeric_scaled = assets['scaler'].transform([raw_numeric_values])
            
           
            encoded_cats = np.zeros((1, len(encoded_cols)))
            encoded_df = pd.DataFrame(encoded_cats, columns=encoded_cols)
            
            
            active_cat_columns = [
                f"Location_{location}",
                f"WindGustDir_{wind_gust_dir}",
                f"WindDir9am_{wind_dir_9am}",
                f"WindDir3pm_{wind_dir_3pm}",
                f"RainToday_{rain_today}"
            ]
            
            
            for col in active_cat_columns:
                if col in encoded_df.columns:
                    encoded_df[col] = 1.0
            
            
            X_processed = np.hstack([numeric_scaled, encoded_df.values])
            
            
            prediction = model.predict(X_processed)[0]
            probabilities = model.predict_proba(X_processed)[0]
            
            
            st.subheader("🎯 Результат прогнозу:")
            rain_probability = probabilities[1] * 100
            
            if prediction == 1 or prediction == "Yes":
                st.error(f" **Так, завтра прогнозується дощ!**")
                
            else:
                st.success(f" **Ні, завтра дощу не передбачається.**")
            st.info(f" Ймовірність опадів: **{rain_probability:.2f}%**")
            
        except Exception as e:
            st.error(f"Помилка під час виконання прогнозу: {e}")