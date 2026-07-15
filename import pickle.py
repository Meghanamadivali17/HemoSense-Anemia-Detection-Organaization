import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)
with open('anemia_detector.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        gender = int(request.form['gender'])
        hemoglobin = float(request.form['hemoglobin'])
        rbc_count = float(request.form['rbc_count']) 
        input_features = np.array([[gender, hemoglobin, rbc_count]])
        prediction = model.predict(input_features)
        result_text = "Anemic Detected - Consult a Doctor" if prediction == 1 else "Normal - You are Healthy!"
        
        return render_template('index.html', prediction_text=f'Result: {result_text}')

if __name__ == '__main__':
    app.run(debug=True)