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
    data={
        "0": "10,000 - 20,000",
        "1": "20,000 - 35,000",
        "2": "30,00 - 70,000",
        "3": "50,000 - 1,50,000",
        "4": "> 1,00,000"
    }
	# Format prediction text for display in "index.html"
    return render_template('result.html', prediction ='Moblile price range should be {}'.format(data.get(str(prediction[0]),"10,00-1,00,000")))

if __name__ == '__main__':
    app.run(debug=False)
