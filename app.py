# import numpy as np
# from flask import Flask, request, jsonify, render_template
# import pickle
# import json

# app = Flask(__name__)
# model = pickle.load(open('model.pkl', 'rb'))

# # Load crop details and translations from JSON
# with open('crop_data.json', 'r', encoding='utf-8') as file:
#     crop_data = json.load(file)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     '''
#     For rendering results on HTML GUI
#     '''
#     # Get input values from the form
#     floatvals=request.form.values()
#     floatvals = list(floatvals)[:-1]
#     int_features = [float(x) for x in floatvals]
#     # int_features = [float(x) for x in request.form.values()[:-1]]
#     final_features = [np.array(int_features)]
    
#     # Make prediction
#     prediction = model.predict(final_features)
#     crop_name = prediction[0]  # Get the predicted crop name
#     # Get user-selected language (default to English)
#     language = request.form.get('language', 'english')
#     # crop_name="coffee"
#     # Get crop details inthe selected language
#     crop_info = crop_data['crops'].get(crop_name, {})
#     crop_details = crop_info.get('details', {}).get(language, "Details not available in this language.")

#     print(crop_name,language,crop_details)
#     # Render the result on the HTML page
#     return render_template('index.html', 
#                            prediction_text=f'{crop_name}',
#                            crop_details=crop_details,
#                            input_values=int_features,
#                            selected_language=language)

# if __name__ == "__main__":
#     app.run(debug=True)
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for
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

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get form data
#         print("DEBUG: Received Form Data ->", request.form)
#         floatvals = list(request.form.values())[:-1]  # Exclude the last non-numeric value
#         int_features = [float(x) for x in floatvals]  # Convert to float
#         final_features = [np.array(int_features)]  # Prepare for model

#         # 🚀 Dummy Prediction (Replace with ML Model)
#         crop_name = "Wheat"

#         # 🚀 Dummy Crop Details
#         crop_details = "Best grown in moderate climate with medium rainfall."

#         # Get selected language
#         language = request.form.get('language', 'english')

#         print("DEBUG: Input Values:", int_features)  # 🛠️ Debugging line

#         return render_template('results.html',
#                                prediction_text=crop_name,
#                                crop_details=crop_details,
#                                input_values=int_features,  # 🚀 Make sure this is passed!
#                                selected_language=language)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not model:
            return jsonify({"error": "Model not loaded. Check 'model.pkl' file."}), 500

        # ✅ Get input values from the form
        form_values = list(request.form.values())[:-1]  # Exclude last non-numeric value
        int_features = [float(x) for x in form_values]
        final_features = np.array([int_features])  # Format input for model

        # 🎯 **Make Prediction**
        prediction = model.predict(final_features)
        crop_name = prediction[0]  # Predicted crop name

        # ✅ Get user-selected language (default: English)
        language = request.form.get('language', 'english')

        # 🌾 **Fetch Crop Details**
        crop_info = crop_data.get('crops', {}).get(crop_name, {})
        crop_details = crop_info.get('details', {}).get(language, "Details not available in this language.")

        print(f"DEBUG: Crop -> {crop_name} | Language -> {language} | Details -> {crop_details}")

        # ✅ Render results
        # return render_template('index.html', 
        #                        prediction_text=f'{crop_name}',
        #                        crop_details=crop_details,
        #                        input_values=int_features,
        #                        selected_language=language)
        return render_template('results.html', 
                               prediction_text=f'{crop_name}',
                               crop_details=crop_details,
                               input_values=int_features,
                               selected_language=language)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/results')
def results():
    '''
    Display the prediction results on a separate page
    '''
    crop_name = request.args.get('crop_name', 'Unknown')
    crop_details = request.args.get('details', 'No details available')
    language = request.args.get('language', 'English')

    return render_template('results.html', 
                           prediction_text=crop_name,
                           crop_details=crop_details,
                           selected_language=language)

if __name__ == "__main__":
    app.run(debug=True)
