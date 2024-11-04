import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

# Create flask app
flask_app = Flask(__name__)
# Tải mô hình
model = joblib.load(open("ada_model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text = "Kết quả dự đoán là (0:Không yêu cầu đền bù, 1:Yêu cầu đèn bù): {}".format(prediction))

if __name__ == "__main__":
    flask_app.run(debug=True)