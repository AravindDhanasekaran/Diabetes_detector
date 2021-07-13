from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('diabetesprediction.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        preg = int(request.form['a'])
        glucose = int(request.form['b'])
        bp = int(request.form['c'])
        st = int(request.form['d'])
        insulin = int(request.form['e'])
        bmi = float(request.form['f'])
        dpf = float(request.form['g'])
        age = int(request.form['h'])
        
        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        pred=model.predict(data)
        
        return render_template('result.html', prediction=pred)

if __name__ == '__main__':
	app.run(debug=True)
