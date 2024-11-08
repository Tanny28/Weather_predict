
# ML endpoint for climate prediction by city and timeframe
from flask import Flask, request, jsonify
import Backend  # Ensure Backend has predict_lstm function
import DL  # Ensure DL has classify_and_predict function

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Prediction API. Use /predict-climate or /predict-image for predictions."

@app.route('/predict-climate', methods=['POST'])
def predict_climate():
    try:
        data = request.json
        city = data.get('city')
        timeframe = data.get('timeframe')
        print(f"Received data - city: {city}, timeframe: {timeframe}")  # Debug output

        if not city or not timeframe:
            return jsonify({"error": "City and timeframe are required"}), 400

        # Check if Backend module has the 'predict_lstm' function
        if not hasattr(Backend, 'predict_lstm'):
            print("predict_lstm function not found in Backend")  # Debug output
            return jsonify({"error": "predict_lstm function not found in Backend module"}), 500

        # Call ML prediction function
        result = Backend.predict_lstm(city, timeframe)
        print(f"Prediction result: {result}")  # Debug output
        return jsonify(result)
    except Exception as e:
        print(f"Error in /predict-climate endpoint: {e}")  # Error details
        return jsonify({"error": "Internal server error"}), 500

# DL endpoint for image-based weather prediction
@app.route('/predict-image', methods=['POST'])
def predict_image():
    image_file = request.files.get('image')
    if not image_file:
        return jsonify({"error": "No image file provided"}), 400
    
    # Call DL prediction function from PUNE_DL.py
    # Replace with the actual function in PUNE_DL.py that handles image input
    result = DL.classify_and_predict(image_file)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
