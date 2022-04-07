from flask import Flask, jsonify, request
from classifier import get_prediction

app = Flask(__name__)

@app.route("/predictData", methods = ["POST"])
def predictData() :
  image = request.files.get("alphabet")
  prediction = get_prediction(image)
  return jsonify({
    "Prediction" : prediction
  }), 200

if __name__ == "__main__" :
  app.run(debug = True)