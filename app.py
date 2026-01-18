from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("house_price_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        features = [
            float(request.form["MedInc"]),
            float(request.form["HouseAge"]),
            float(request.form["AveRooms"]),
            float(request.form["AveBedrms"]),
            float(request.form["Population"]),
            float(request.form["AveOccup"]),
            float(request.form["Latitude"]),
            float(request.form["Longitude"]),
        ]

        prediction = model.predict([features])[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)