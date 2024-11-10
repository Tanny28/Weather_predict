import streamlit as st
from PIL import Image
import os
import Backend  # Importing functions from Backend.py
import DL  # Importing functions from DL.py

# Setting up custom CSS for styling
st.markdown("""
    <style>
    /* Background styling */
    .main {
        background-image: url('https://images.unsplash.com/photo-1561484936-cdf71e7b7abf');
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        color: white;
    }
    /* Card style for prediction results */
    .card {
        padding: 20px;
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
        color: #333;
    }
    /* Button animations */
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
# Embed Canva Design with Larger Window
st.components.v1.html("""
<div style="position: relative; width: 100%; height: 0; padding-top: 80%; padding-bottom: 0; box-shadow: 0 2px 8px 0 rgba(63,69,81,0.16); margin-top: 1.6em; margin-bottom: 0.9em; overflow: hidden; border-radius: 8px; will-change: transform;">
  <iframe loading="lazy" style="position: absolute; width: 100%; height: 100%; top: 0; left: 0; border: none; padding: 0; margin: 0;"
    src="https://www.canva.com/design/DAGWE1U365Q/x6tpJTuRxKPvXSxYUBZnmw/watch?embed" allowfullscreen="allowfullscreen" allow="fullscreen">
  </iframe>
</div>
""", height=500)  # Adjusting the height for better full view


# Define a function to predict weather based on city and timeline
def city_based_prediction(city, timeline):
    try:
        prediction = Backend.get_prediction(city, timeline)
        return prediction
    except ValueError as e:
        return str(e)

# Define a function to predict weather based on an uploaded image
def image_based_prediction(image_path):
    prediction = DL.predict_from_image(image_path)
    return prediction

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
        
        # Format result as a structured display with styled cards
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
uploaded_image = st.file_uploader("Upload an Image of the Climate", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    image_path = "temp_image.jpg"
    image.save(image_path)

    if st.button("Predict by Image"):
        result = image_based_prediction(image_path)
        st.markdown("#### Weather Prediction based on Image Analysis")

        # Display result in a structured format with cards
        if isinstance(result, dict):
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.write(f"**Temperature:** {result['temperature']} °C")
            st.write(f"**Humidity:** {result['humidity']} %")
            st.write(f"**Rainfall:** {result['rainfall']} mm")
            st.write(f"**Wind Speed:** {result['wind_speed']} km/h")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error(result)

        os.remove(image_path)  # Remove temporary image after prediction
