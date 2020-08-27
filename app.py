import numpy as np
from flask import Flask,render_template,request
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template("index.html")
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict',methods = ['GET','POST'])
def predict():
    
    int_features = [int(x) for x in request.form.values()]
    #df = pd.DataFrame[int_features]
    #pca = PCA()
    final_features= np.array(int_features)
    pca_scaled = pca.fit(final_features)
    pca_scaled = pca.transform(final_features)
    prediction = model.predict(pca_scaled)
    
    if(prediction == 0):
        val_predict = "Patient is not Diabetic"
    else:
        val_predict = "patient is Diabetic"
    
    return render_template('index.html',prediction_text = val_predict)


if __name__ == "__main__":
    app.run(debug = False)


