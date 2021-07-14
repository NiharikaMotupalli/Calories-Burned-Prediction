import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from imblearn.over_sampling import SMOTE
import pickle
import warnings 
warnings.simplefilter("ignore")

def remove_outlier(dataset):
    Q1 = dataset.quantile(0.25)
    Q3 = dataset.quantile(0.75)
    IQR = Q3 - Q1
    print(IQR)
    dataset=dataset[~((dataset<(Q1 - 1.5 * IQR)) | (dataset > (Q3 + 1.5 * IQR))).any(axis=1)]
    print(dataset) 
    
#Smote function
def smote_func(dataset):
    x=dataset.iloc[:,0:-1].values
    y=dataset.iloc[:,-1].values
    os=SMOTE(random_state=20)
    x_train,x_test,y_train,y_test = train_test_split(x , y , test_size=0.2,random_state=10)
    os_data_x,os_data_y=os.fit_sample(x_train,y_train)
    return os_data_x,x_test, os_data_y,y_test  
    
#Algorithms.
#DecisionTreeClassifier  
'''def DTC(dataset,smote):
    DTC_obj=DecisionTreeClassifier()
    if smote=='True':
        x_train,x_test,y_train,y_test=smote_func(dataset)
    else:
        x_train,x_test,y_train,y_test = train_test_split(x , y , test_size=0.2,random_state=10)      
    DTC_obj.fit(x_train,y_train)
    #Predict the outcome
    y_predict_DTC=DTC_obj.predict(x_test)
    print(classification_report(y_test,y_predict_DTC))
    print("Accuracy percentage DTC:"+"{:.2f}".format(accuracy_score(y_test,y_predict_DTC)*100))
    b=accuracy_score(y_test,y_predict_DTC)
    #Confusion matrix
    cm = confusion_matrix(y_test, y_predict_DTC)
    print(cm)
    return b'''

calories = pd.read_csv('D:/projects/IV-II (Calories burned prediction - ML)/DatasetExcel files/calories.csv')
#calories.head()
exercise = pd.read_csv('D:/projects/IV-II (Calories burned prediction - ML)/DatasetExcel files/exercise.csv')
#exercise.head()
dataset = pd.merge(exercise, calories, on = 'User_ID')
dataset.head()

#dataset.info()
dataset.isnull()
dataset.hist()
print(dataset.isnull().sum())


dataset.describe()

dataset['Gender'] = dataset['Gender'].map({"male":0,"female":1 });

remove_outlier(dataset)

x=dataset.iloc[:,0:-1].values
y=dataset.iloc[:,-1].values
x_train,x_test,y_train,y_test = train_test_split(x , y , test_size=0.2,random_state=10)

y=dataset['Calories']
x=dataset.drop(['Calories'],axis=1)

'''os=SMOTE(random_state=20)
os_data_x,os_data_y=os.fit_resample(x,y)
pd.DataFrame(data=os_data_x,columns=x.columns)
pd.DataFrame(data=os_data_y,columns=['Calories'])'''
x_train,x_test,y_train,y_test = train_test_split(x , y , test_size=0.20,random_state=10)


dtc=DecisionTreeClassifier()

#Train the model
dtc.fit(x_train,y_train)

# Saving model to disk
pickle.dump(dtc, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
