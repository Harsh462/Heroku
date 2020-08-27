import numpy as np
from flask import Flask,render_template,request
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template("index.html")
@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/predict',methods = ['GET','POST'])
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
        
           data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
           pca = PCA()
           data1=pca.fit(data)
           data2 = pca.transform(data)
           my_prediction = model.predict(data2)
    
        
           
    
           return render_template('index.html',prediction_text = my_prediction)


if __name__ == "__main__":
    app.run(debug = False)


