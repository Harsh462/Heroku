import numpy as np
from flask import Flask,render_template,request
import pickle
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression


model = pickle.load/(open('model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")
@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/predict',methods = ['POST'])
def predict():
    if request.method == 'POST':
           
           preg = int(request.form['pregnancies'])
           glucose = int(request.form['glucose'])
           bp = int(request.form['bloodpressure'])
           st = int(request.form['skinthickness'])
           insulin = int(request.form['insulin'])
           bmi = float(request.form['bmi'])
           dpf = float(request.form['dpf'])
           age = int(request.form['age'])
        
           data = pd.DataFrame([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
           
           data = normalize(data)
           
           my_prediction = model.predict(data)
           
          
           if (my_prediction)== 0:
                   val_predict = "Patient is not Diabetic"
           else:
                   val_predict = "patient is Diabetic"
    
        
           
    
           return render_template('index.html',prediction_text = val_predict)


if __name__ == "__main__":
    app.run(debug = False)


