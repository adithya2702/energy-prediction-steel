import pickle
import pandas as pd
from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

# Load the model files into a dictionary
models = {}
for i in range(1, 9):
    model_file_path = f'Model{i}.pkl'
    if os.path.exists(model_file_path):
        with open(model_file_path, 'rb') as model_file:
            models[f'Model{i}'] = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html', model_options=list(models.keys()))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    user_input = data['user_input']
    selected_model = data['selected_model']
    
    # Get the selected model from the dictionary
    selected_model = models[selected_model]
    
    user_datetime = pd.to_datetime(user_input)
    user_day_of_week = user_datetime.dayofweek
    user_month = user_datetime.month
    user_hour = user_datetime.hour
    predicted_energy_consumption = selected_model.predict([[user_day_of_week, user_month, user_hour]])
    result = {
        'predicted_consumption': predicted_energy_consumption[0]
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
