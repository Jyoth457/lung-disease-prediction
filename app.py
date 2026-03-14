from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

data = pd.read_csv("lung_disease_dataset_1000_records.csv")

X = data.drop("Lung_Disease", axis=1)
y = data["Lung_Disease"]

model = RandomForestClassifier()
model.fit(X, y)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    age = int(request.form['age'])
    smoking = int(request.form['smoking'])
    pollution = int(request.form['pollution'])
    cough = int(request.form['cough'])
    breath = int(request.form['breath'])
    pain = int(request.form['pain'])
    fatigue = int(request.form['fatigue'])
    wheezing = int(request.form['wheezing'])

    patient = np.array([[age, smoking, pollution, cough, breath, pain, fatigue, wheezing]])

    prediction = model.predict(patient)

    if prediction[0] == 1:
        result = "Lung Disease Detected"
    else:
        result = "No Lung Disease"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)