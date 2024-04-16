import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from flask import Flask,render_template,request
app=Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict_data",methods=['GET','POST'])
def predict_datapoint():
   if request.method=="GET":
        return render_template("home.html")
   else:
        data=CustomData(
                age=request.form['age'],
                sex=request.form['sex'],
            	bmi=request.form['bmi'],
                children=request.form['childern'],
                smoker=request.form['smoker'],
                region=request.form['region'],
        )
        # print("Before Prediction")    
        pred_df=data.get_as_dataframe()
        # print(pred_df)
        
        
        predict_pipeline=PredictPipeline()
        # print("After Prediction")

        results=predict_pipeline.predict(pred_df)
        results=np.round(results,2)
        # print(results)
        return render_template('home.html',results=results[0])

if __name__=="__main__":
    app.run(host="0.0.0.0",port=8080,debug=True)

#URL:http://127.0.0.1:8080/predict_data