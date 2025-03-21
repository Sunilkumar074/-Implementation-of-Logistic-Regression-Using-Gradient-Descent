# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Load the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary. 6.Define a function to predict the 
   Regression value.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SUNIL KUMAR P.B.
RegisterNumber:  212223040213
*/


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("C:/Users/admin/Downloads/Placement_Data.csv")
dataset

dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes

dataset

X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values

Y

theta=np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1 /(1+np.exp(-z))

def loss(theta,X,y):
    h=sigmoid(x.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient=X.T.dot(h-y)/m
        theta-=alpha*gradient
    return theta

theta=gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)

def predict(theta,X):
    h=sigmoid(X.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred 

y_pred=predict(theta,X)

accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)

print(y_pred)

print(Y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```



## Output:
### Dataset
![image](https://github.com/user-attachments/assets/0a69f4fd-78f6-4a12-b28e-5492bf41f90c)
### Dataset.dtypes
![image](https://github.com/user-attachments/assets/132b7635-1494-46b4-b3d1-8dba39776200)
### Dataset
![image](https://github.com/user-attachments/assets/a5ff9b2b-ceb5-4163-9ec1-2df7b341d50a)
### Y
![image](https://github.com/user-attachments/assets/4893b56d-007a-42d3-922a-3482e82fec0c)
### Accuracy
![image](https://github.com/user-attachments/assets/d785e48a-caba-4196-b0e9-7049f75b19a4)
### Y_pred
![image](https://github.com/user-attachments/assets/6922b97e-c712-443a-8a62-a41875385485)
### Y
![image](https://github.com/user-attachments/assets/6e9361b6-7615-4f6c-97b5-210a8a7d1790)
### Y_prednew
![image](https://github.com/user-attachments/assets/ca9e27a5-4147-4459-8fa4-6097bded994f)
### Y_prednew
![image](https://github.com/user-attachments/assets/b243668c-5103-4f5d-a0aa-3f78d71f076b)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
