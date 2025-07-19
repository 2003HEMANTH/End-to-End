from flask import Flask, request, render_template
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict_data():
    if request.method == 'POST':
        try:
            # ✅ Fetch input from form safely with default fallback
            data = CustomData(
                bedrooms=float(request.form.get('bedrooms', 3) or 3),
                bathrooms=float(request.form.get('bathrooms', 2) or 2),
                livingArea=float(request.form.get('livingArea', 1200) or 1200),
                price=float(request.form.get('price', 250000) or 250000),
                rentZestimate=float(request.form.get('rentZestimate', 1500) or 1500),
                pageViewCount=float(request.form.get('pageViewCount', 0) or 0),
                favoriteCount=float(request.form.get('favoriteCount', 0) or 0),
                propertyTaxRate=float(request.form.get('propertyTaxRate', 1.2) or 1.2),
                timeOnZillow=float(request.form.get('timeOnZillow', 48) or 48),
                yearBuilt=float(request.form.get('yearBuilt', 2005) or 2005),
                homeStatus=request.form.get('homeStatus', "FOR_SALE") or "FOR_SALE",
                homeType=request.form.get('homeType', "Single Family") or "Single Family",
                city=request.form.get('city', "San Jose") or "San Jose",
                zipcode=request.form.get('zipcode', "95123") or "95123",
                state=request.form.get('state', "CA") or "CA"
            )

            input_df = data.get_data_as_data_frame()

            predict_pipeline = PredictPipeline()
            prediction = predict_pipeline.predict(input_df)

            output = round(prediction[0], 2)

            return render_template("home.html", results=output)

        except Exception as e:
            return render_template("home.html", results=None, error=str(e))

    return render_template("home.html", results=None)

if __name__ == "__main__":
    app.run(debug=True)
