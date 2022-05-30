# -*- coding: utf-8 -*-
"""
Created on Sun May 29 19:04:52 2022

@author: Harish
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
from tensorflow.keras.models import load_model

model=load_model('airpassengers.h5')
sc=joblib.load('scaler')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login',methods=['POST'])
def predict():
    yr='yr'
    mn='mn'
    dy='dy'
    yr=request.form[yr]
    mn=request.form[mn]
    dy=request.form[dy]
    data=[[float(yr),float(mn),float(dy)]]
    data=sc.transform(data)
    prediction=model.predict(data)
    output='Predicted price is '+str(round(prediction[0][0],2))
    return render_template('index.html', value=output)

if __name__ == "__main__":
    app.run(debug=True)
