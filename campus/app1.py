import numpy as np

from flask import Flask, render_template, request
import pickle

app = Flask(__name__,template_folder="templates")
model = pickle.load(open('decision.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ssc_percentage = float(request.form['ssc'])
    hsc_percentage = float(request.form['hsc'])
    stream = request.form['stream']
    work_experience = int(request.form['work_experience'])
    etest_percentage = float(request.form['etest'])
    specialization = request.form['specialization']
    mba_percentage = float(request.form['mba'])

    arr = np.array([ssc_percentage,hsc_percentage,stream,work_experience,etest_percentage,specialization, mba_percentage])
    brr = np.asarray(arr, dtype=float)
    output = model.predict([brr])
    if(output==1):
        out = 'You have high chances of getting placed!!!'
    else:
        out = 'You have low chances of getting placed. All the best.'
    return render_template('out.html', output=out)

if __name__ == '__main__':
    app.run(debug=True)
