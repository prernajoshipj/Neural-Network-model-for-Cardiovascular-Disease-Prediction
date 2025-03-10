
from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

FASTAPI_URL = "http://127.0.0.1:8000/predict"  # URL of your FastAPI backend

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Get user input from form
        user_data = {
            "age": int(request.form["age"]),
            "gender": int(request.form["gender"]),
            "height": int(request.form["height"]),
            "weight": float(request.form["weight"]),
            "ap_hi": int(request.form["ap_hi"]),
            "ap_lo": int(request.form["ap_lo"]),
            "smoke": int(request.form["smoke"]),
            "alco": int(request.form["alco"]),
            "active": int(request.form["active"]),
            "gluc_2": int(request.form["gluc_2"]),
            "gluc_3": int(request.form["gluc_3"]),
            "cholesterol_2": int(request.form["cholesterol_2"]),
            "cholesterol_3": int(request.form["cholesterol_3"])
        }

        # Send request to FastAPI
        response = requests.post(FASTAPI_URL, json=user_data)
        result = response.json()

        return render_template("index.html", result=result["cardiovascular_risk"])

    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)
