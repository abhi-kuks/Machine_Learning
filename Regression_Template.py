# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 23:23:52 2019

@author: Abhishek
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

dataset=pd.read_csv("Position_Salaries.csv")
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

"""from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,
											   random_state=0)"""

#Fitting the Regression Model  to the dataset

#predicting the salary using Polynomial Regression Model
y_pred=regressor.predict(6.5) 
   
#Visualising the Regression Results

plt.plot(x,regressor.predict(x)),color="blue")
plt.title("Truth or Bluff (Regression Model)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()


#Visualising the Regression Results(for more smoother and high resolution Curve)
x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color="red")
plt.plot(x_grid,regressor.predict(x_grid),color="blue")
plt.title("Truth or Bluff (Regression Model)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()