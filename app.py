from flask import Flask, render_template, request
from predict import CarPredictor

app = Flask(__name__)
predictor = CarPredictor()

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        input_data = {
            "year": int(request.form["year"]),
            "present_price": float(request.form["present_price"]),
            "kms_driven": int(request.form["kms_driven"]),
            "seller_type": request.form["seller_type"],
            "transmission": request.form["transmission"],
            "owner": int(request.form["owner"]),
            "car_name": request.form.get("car_name", "unknown"),
            "selling_price": 0  # dummy value (not used in prediction)
        }

        prediction = predictor.predict(input_data)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
