import streamlit as st
from PIL import Image
import os
import Backend  # Importing functions from Backend.py
import DL  # Importing functions from DL.py
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt
import requests  # For the chatbot integration


LARAVEL_CHATBOT_API_URL = 'https://api.vultrinference.com/v1/chat/completions'

API_KEY = '4BR3W4SABFO4SMJWNVWSOCPV3LOVIVZ7DZAQ' 

def send_message_to_chatbot(message):
    headers = {
        'Authorization': f'Bearer {API_KEY}',  # Assuming the API uses Bearer token authentication
        'Content-Type': 'application/json'
    }
    payload = {
        "model": "llama2-13b-chat-Q5_K_M",  # Model specified as per the PHP example
        "messages": [
            {
                "role": "user",
                "content": message
            }
        ],
        "max_tokens": 150,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(LARAVEL_CHATBOT_API_URL, json=payload, headers=headers)
        if response.status_code == 200:
            return response.json().get('choices')[0].get('message', {}).get('content', 'Error: No response from chatbot')
        else:
            return f"Error: {response.status_code} - {response.json().get('message', response.text)}"
    except requests.exceptions.RequestException as e:
        return f"Exception: {str(e)}"

# Setting up custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-image: url('https://images.unsplash.com/photo-1561484936-cdf71e7b7abf');
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        color: white;
    }
    .card {
        padding: 20px;
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
        color: #333;
    }
    .stButton>button {
        color: #fff;
        background-color: #4CAF50;
        font-size: 18px;
        transition: transform 0.2s ease;
    }
    .stButton>button:hover {
        transform: scale(1.1);
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

# Embed Canva Design
st.components.v1.html("""
<div style="position: relative; width: 100%; height: 0; padding-top: 80%; padding-bottom: 0; box-shadow: 0 2px 8px 0 rgba(63,69,81,0.16); margin-top: 1.6em; margin-bottom: 0.9em; overflow: hidden; border-radius: 8px; will-change: transform;">
  <iframe loading="lazy" style="position: absolute; width: 100%; height: 100%; top: 0; left: 0; border: none; padding: 0; margin: 0;"
    src="https://www.canva.com/design/DAGWE1U365Q/x6tpJTuRxKPvXSxYUBZnmw/watch?embed" allowfullscreen="allowfullscreen" allow="fullscreen">
  </iframe>
</div>
""", height=500)

# Step 1: Define weather categories and classification model
weather_categories = ["humid", "sunny", "foggy", "rainy"]
num_classes = len(weather_categories)

def build_classification_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

classification_model = build_classification_model()

# Step 2: Generate weather data and LSTM model
def generate_weather_data(num_days):
    base_temp = random.randint(10, 30)
    temps = base_temp + 5 * np.sin(np.linspace(0, 3 * np.pi, num_days)) + np.random.normal(0, 2, num_days)
    dates = pd.date_range(start="2023-01-01", periods=num_days)
    data = pd.DataFrame({"date": dates, "temperature": temps})
    return data

data = generate_weather_data(365)
train_data = data['temperature'].values

def prepare_lstm_data(data, look_back=30):
    generator = TimeseriesGenerator(data, data, length=look_back, batch_size=1)
    return generator

def train_lstm_model(train_data):
    look_back = 30
    lstm_train_gen = prepare_lstm_data(train_data, look_back=look_back)
    
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, activation='relu', input_shape=(look_back, 1)))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(lstm_train_gen, epochs=5, verbose=1)
    
    return lstm_model

lstm_model = train_lstm_model(train_data)

def predict_lstm(model, data, look_back=30, days=30):
    predictions = []
    current_batch = data[-look_back:]
    current_batch = current_batch.reshape((1, look_back, 1))
    
    for _ in range(days):
        pred = model.predict(current_batch)[0]
        predictions.append(pred)
        current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)
    
    return np.array(predictions).flatten()

# Step 3: Classify and predict function
def classify_and_predict(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = classification_model.predict(img_array)
    class_index = np.argmax(predictions)
    weather_type = weather_categories[class_index]
    
    st.write(f"Classified weather condition: {weather_type}")
    
    periods = {"1 month": 30, "6 months": 180, "1 year": 365, "2 years": 730}
    for period_name, days in periods.items():
        lstm_predictions = predict_lstm(lstm_model, train_data, days=days)
        
        actual_data = pd.DataFrame({
            "Days": range(len(train_data)),
            "Temperature": train_data,
            "Type": "Actual"
        })
        predicted_data = pd.DataFrame({
            "Days": range(len(train_data), len(train_data) + days),
            "Temperature": lstm_predictions,
            "Type": f"Prediction ({period_name})"
        })
        
        plt.figure(figsize=(10, 5))
        plt.plot(actual_data["Days"], actual_data["Temperature"], label="Actual")
        plt.plot(predicted_data["Days"], predicted_data["Temperature"], label=f"Prediction ({period_name})", linestyle="--")
        plt.title(f"Temperature Prediction for {period_name}")
        plt.xlabel("Days")
        plt.ylabel("Temperature")
        plt.legend()
        
        st.pyplot(plt)
        plt.close()

# City-based prediction function
def city_based_prediction(city, timeline):
    try:
        prediction = Backend.get_prediction(city, timeline)
        return prediction
    except ValueError as e:
        return str(e)

# Streamlit App Layout
st.title("🌦️ Weather Prediction App")
st.markdown("<h3 style='text-align: center;'>Get Detailed Weather Forecasts by City or through Image Analysis</h3>", unsafe_allow_html=True)

# City-based Prediction Section
st.subheader("🌆 City-based Weather Prediction")
st.markdown("<div class='card'>Enter city name and select timeline for weather forecast</div>", unsafe_allow_html=True)

city = st.text_input("Enter City Name")
timeline = st.selectbox("Select Prediction Timeline", ["1 month", "6 months", "1 year", "2 years"])

if st.button("Predict by City"):
    if city:
        result = city_based_prediction(city, timeline)
        st.markdown(f"#### Weather Prediction for {city.capitalize()} over {timeline}")
        
        if isinstance(result, dict):
            for model, data in result.items():
                st.markdown(f"<div class='card'><strong>{model} Model Results:</strong>", unsafe_allow_html=True)
                st.write(f"**Temperature:** {data['temperature']} °C")
                st.write(f"**Humidity:** {data['humidity']} %")
                st.write(f"**Rainfall:** {data['rainfall']} mm")
                st.write(f"**Wind Speed:** {data['wind_speed']} km/h")
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error(result)
    else:
        st.error("Please enter a city name.")

# Image-based Prediction Section
st.subheader("📷 Image-based Weather Prediction")
st.markdown("<div class='card'>Upload an image to predict weather conditions through AI analysis</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload an Image of the Climate", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    temp_image_path = f"temp_{uploaded_file.name}"
    with open(temp_image_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    classify_and_predict(temp_image_path)
    
    os.remove(temp_image_path)

# Chatbot Section
st.subheader("💬 Chat with our Weather Bot")
st.markdown("<div class='card'>Ask the weather bot about weather conditions, predictions, or any queries related to climate.</div>", unsafe_allow_html=True)

user_message = st.text_input("Your message to the bot")

if st.button("Send Message"):
    if user_message:
        bot_response = send_message_to_chatbot(user_message)
        st.markdown(f"**Bot Response:** {bot_response}")
    else:
        st.error("Please enter a message to chat with the bot.")
