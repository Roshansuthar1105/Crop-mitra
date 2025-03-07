import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import json

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

# Load crop details and translations from JSON
with open('crop_data.json', 'r', encoding='utf-8') as file:
    crop_data = json.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # Get input values from the form
    floatvals=request.form.values()
    floatvals = list(floatvals)[:-1]
    int_features = [float(x) for x in floatvals]
    # int_features = [float(x) for x in request.form.values()[:-1]]
    final_features = [np.array(int_features)]
    
    # Make prediction
    prediction = model.predict(final_features)
    crop_name = prediction[0]  # Get the predicted crop name
    # Get user-selected language (default to English)
    language = request.form.get('language', 'english')
    # crop_name="coffee"
    # Get crop details inthe selected language
    crop_info = crop_data['crops'].get(crop_name, {})
    crop_details = crop_info.get('details', {}).get(language, "Details not available in this language.")

    print(crop_name,language,crop_details)
    # Render the result on the HTML page
    return render_template('index.html', 
                           prediction_text=f'{crop_name}',
                           crop_details=crop_details,
                           input_values=int_features,
                           selected_language=language)

if __name__ == "__main__":
    app.run(debug=True)