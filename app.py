
import numpy as np
from flask import Flask, request, render_template,jsonify
import pickle
import json

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

# Load crop details from JSON
with open('crop_info.json', 'r', encoding='utf-8') as file:
    crop_data = json.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not model:
            return jsonify({"error": "Model not loaded. Check 'model.pkl' file."}), 500

        # Get input values from the form
        # form_values = list(request.form.values())[:-1]  # Exclude last non-numeric value
        form_values = list(request.form.values())  # Exclude last non-numeric value
        int_features = [float(x) for x in form_values]
        final_features = np.array([int_features])  # Format input for model

        # Make Prediction
        prediction = model.predict(final_features)
        crop_name = prediction[0]  # Predicted crop name

        # Get user-selected language (default: English)
        language = request.form.get('language', 'english')

        # Fetch Crop Details
        crop_info = crop_data.get('crops', {}).get(crop_name, {})

        if not crop_info:
            return render_template('results.html', 
                                   prediction_text=crop_name,
                                   crop_details={"error": "No details available for this crop."},
                                   selected_language=language)

        # Render results page
        return render_template('results.html', 
                               prediction_text=crop_name,
                               crop_details=crop_info,
                               selected_language=language)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)