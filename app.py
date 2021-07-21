import numpy as np
import pickle

from flask import Flask,request,render_template
app = Flask(__name__)
model = pickle.load(open('model/model.pkl', 'rb'))

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

   Gender=request.form['Gender']   
   values.append(Gender)
   
   Age=request.form['Age']
   values.append(Age)
   
   Weight=request.form['Weight']
   values.append(Weight)

   Duration=request.form['Duration']
   values.append(Duration)

   Heart_Rate=request.form['Heart_Rate']
   values.append(Heart_Rate)

   final_values=[np.array(values)]
   print(final_values)
   
   result= model.predict(final_values)

   output= str(result[0])
  
   return render_template('result.html',User_ID=User_ID,Gender=Gender,Age=Age,Weight=Weight,Duration=Duration,Heart_Rate=Heart_Rate,Calories=output)
  

if __name__ == '__main__':
   app.run(debug=True,use_reloader=False)
