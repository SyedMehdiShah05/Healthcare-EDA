from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('healthcare_model.pkl', 'rb') as file:
    model = pickle.load(file)

# The same dictionaries we used in Colab
gender_map = {'Male': 0, 'Female': 1}
blood_type_map = {'B-': 0, 'A+': 1, 'A-': 2, 'O+': 3, 'AB+': 4, 'AB-': 5, 'B+': 6, 'O-': 7}
medical_condition_map = {'Cancer': 0, 'Obesity': 1, 'Diabetes': 2, 'Asthma': 3, 'Hypertension': 4, 'Arthritis': 5}
admission_type_map = {'Urgent': 0, 'Emergency': 1, 'Elective': 2}
medication_map = {'Paracetamol': 0, 'Ibuprofen': 1, 'Aspirin': 2, 'Penicillin': 3, 'Lipitor': 4}

# Reverse mapping for the target result
result_map = {0: 'Normal', 1: 'Inconclusive', 2: 'Abnormal'}


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from the form
        age = int(request.form['age'])
        gender = gender_map[request.form['gender']]
        blood_type = blood_type_map[request.form['blood_type']]
        condition = medical_condition_map[request.form['medical_condition']]
        admission = admission_type_map[request.form['admission_type']]
        medication = medication_map[request.form['medication']]

        # Create the input array
        input_data = np.array([[age, gender, blood_type, condition, admission, medication]])

        # Make the prediction
        prediction = model.predict(input_data)[0]
        final_result = result_map[prediction]

        return render_template('index.html', prediction_text=f'Predicted Test Result: {final_result}')


if __name__ == "__main__":
    app.run(debug=True)