import numpy as np
import pandas as pd
from flask import Flask,render_template,request
import pickle

from sklearn.linear_model import LogisticRegression


model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)



@app.route('/')
def home():
    return render_template("index.html")
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict',methods = ['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
        
    data = np.array([features])

   
           my_prediction = model.predict(data)
           
           if (my_prediction)== 0:
                   val_predict = "Patient is not Diabetic"
           else:
                   val_predict = "patient is Diabetic"
           
           return render_template('index.html',prediction_text = val_predict)


if __name__ == "__main__":
    app.run(debug = False)


