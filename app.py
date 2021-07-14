import numpy as np
import pickle

from flask import Flask, redirect, url_for,jsonify, request,render_template
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
   return render_template("index.html")

@app.route('/ff',methods = ['POST', 'GET'])
def ff():
   return render_template("form.html")

@app.route('/predict',methods = ['POST'])
def predict():
   values=[]
   
   User_ID=request.form['User_ID']
   values.append(User_ID)
   
   Age=request.form['Age']
   values.append(Age)
   
   Height=request.form['Height']
   values.append(Height)

   Weight=request.form['Weight']
   values.append(Weight)

   Duration=request.form['Duration']
   values.append(Duration)

   Heart_Rate=request.form['Heart_Rate']
   values.append(Heart_Rate)

   Body_Temp=request.form['Body_Temp']
   values.append(Body_Temp)

   Gender=request.form['Gender']
   values.append(Gender)

   final_values=[np.array(values)]
   print(final_values)
   
   result= model.predict(final_values)

   output= result[0]
   print(output)
   
   if output == 0:
      return {'message': User_ID + 'does not burned Calories'}
   else:
      return {'message':User_ID + ' burned Calories'}
  

if __name__ == '__main__':
   app.run(debug=True,use_reloader=False)