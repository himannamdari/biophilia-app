import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Simulate Data
def simulate_data(n_samples=1000):
    np.random.seed(42)
    time_outside = np.random.normal(60, 20, n_samples).clip(0)
    steps_outdoors = np.random.normal(6000, 2000, n_samples).clip(0)
    screen_time = np.random.normal(300, 60, n_samples).clip(60, 600)
    near_nature = np.random.binomial(1, 0.5, n_samples)
    weather_quality = np.random.uniform(0.2, 1.0, n_samples)
    mood_score = np.random.uniform(3, 10, n_samples)

    biophilia_score = (
        0.3 * (time_outside / 120) +
        0.2 * (steps_outdoors / 10000) +
        0.15 * near_nature +
        0.1 * weather_quality +
        0.15 * (mood_score / 10) -
        0.2 * (screen_time / 600)
    ) * 100

    biophilia_score = biophilia_score.clip(0, 100)

    df = pd.DataFrame({
        'time_outside': time_outside,
        'steps_outdoors': steps_outdoors,
        'screen_time': screen_time,
        'near_nature': near_nature,
        'weather_quality': weather_quality,
        'mood_score': mood_score,
        'biophilia_score': biophilia_score
    })
    return df

# Train model and return
@st.cache_data
def train_model():
    df = simulate_data()
    X = df.drop('biophilia_score', axis=1)
    y = df['biophilia_score']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

# Streamlit App
st.title("🌿 BioPulse: Your Biophilia Score Predictor")
st.write("Estimate how connected you are to nature based on your daily habits.")

# User Inputs
time_outside = st.slider("Time spent outside (minutes/day)", 0, 180, 60)
steps_outdoors = st.slider("Steps taken outdoors", 0, 20000, 6000)
screen_time = st.slider("Screen time (minutes/day)", 60, 600, 300)
near_nature = st.selectbox("Do you live near a green space?", ["Yes", "No"])
weather_quality = st.slider("Weather quality today (0 - worst, 1 - best)", 0.0, 1.0, 0.8)
mood_score = st.slider("Your mood today (1 - low, 10 - great)", 1, 10, 7)

# Preprocess Input
user_data = pd.DataFrame({
    'time_outside': [time_outside],
    'steps_outdoors': [steps_outdoors],
    'screen_time': [screen_time],
    'near_nature': [1 if near_nature == "Yes" else 0],
    'weather_quality': [weather_quality],
    'mood_score': [mood_score]
})

# Predict
if st.button("Predict My Biophilia Score"):
    score = model.predict(user_data)[0]
    st.subheader(f"Your estimated Biophilia Score is: {score:.2f} / 100")

    if score < 40:
        st.warning("⚠️ Try spending more time outdoors or near nature to boost your wellbeing!")
    elif score < 70:
        st.info("🙂 You're doing okay—some small changes could make a big difference.")
    else:
        st.success("🌱 Great job! You're well-connected to the natural world.")
