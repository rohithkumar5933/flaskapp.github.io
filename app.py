# Import the libraies
import numpy as np
from flask import Flask, request, render_template
import joblib

# Create the Flask app and load the trained model
app = Flask(__name__)
model = joblib.load('models/trained_mobile_svm_updated_final.pkl')

# Define the '/' root route to display the content from index.html
@app.route('/')
def home():
    return render_template('index.html')

# Define the '/predict' route to:
# - Get form data and convert them to float values
# - Convert form data to numpy array
# - Pass form data to model for prediction

@app.route('/predict',methods=['POST'])
def predict():

    form_data = [x for x in request.form.values()]
    features = [np.array(form_data)]
    prediction = model.predict(features)

    text_map={}
    text_map[0]="should not bemore than 10,000 Rs, less than 10,000 obvious price would be 7000 but might consider upto 10,000"
    text_map[1]="should not be more than 50,000, usually will come under 30 thousand but, depends between 10,000 - 49,999 based on brand"
    return render_template('result.html', prediction =f"{prediction[0]}")

if __name__ == '__main__':
    app.run(debug=False)
