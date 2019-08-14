import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#collecting x and y value

data= pd.read_csv('Salary_Data.csv')

Y= data['Salary'].values
X= data['YearsExperience'].values


# splitiing dataset into training set test set

from sklearn.model_selection import  train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)


#Fitting simple linear regression to the training set
from sklearn.linear_model import  LinearRegression

X_train= X_train.reshape(-1,1)
Y_train=Y_train.reshape(-1,1)
X_test=X_test.reshape(-1,1)
r= LinearRegression()
r.fit(X_train,Y_train)


#predicting the test set result

y_pred= r.predict(X_test)

plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,r.predict(X_train) , color='blue')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.legend()
plt.show()



plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,r.predict(X_train) , color='blue')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.legend()
plt.show()


