from flask import Flask, request, jsonify
from classifier import get_prediction

app = Flask(__name__)

@app.route("/predict-digit", methods = ["POST"])
def predict_digit():
    image = request.files.get("digit")
    prediction = get_prediction(image)
    return jsonify({
        "resultPrediction":prediction
    }),200

if __name__ == "__main__":
    app.run(debug = True)