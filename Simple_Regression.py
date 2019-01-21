# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 17:44:46 2018

@author: Ab_cooks 
"""

#importing the libraries

import numpy as np
import matplotlib.pyplot as plt￼
import pandas as pd 

#importing the dataset
dataset=pd.read_csv("Salary_data.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

#Splitting the dataset in train and test set
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,
											   random_state=0)

#fitting the train set onto the dataset
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)


#predicting the values
y_pred=regressor.predict(x_test)

#Visualizing the data by plotting it(Training set)
plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("Salary vs Years of Experience(Training set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()



#Visualizing the data by plotting it(Testing set)
plt.scatter(x_test,y_test,color="red")
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("Salary vs Years of Experience(Training set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()


 






















