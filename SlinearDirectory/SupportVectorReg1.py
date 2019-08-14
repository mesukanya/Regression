import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset=pd.read_csv("/home/admin1/Desktop/SlinearRegresstion/SlinearDirectory/Position_Salaries.csv")

X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:,2].values
print("before reshaping x",X)

X=X.reshape(-1,1)
Y=Y.reshape(-1,1)

print("after reshaping ",X)

from sklearn.preprocessing import StandardScaler

SC_X= StandardScaler()
SC_Y= StandardScaler()
X= SC_X.fit_transform(X)
Y=SC_Y.fit_transform(Y)

from sklearn.svm import SVR
reg=SVR(kernel='rbf')
reg.fit(X,Y)

#y_pred=reg.predict(X)



#visualising the svr results

plt.scatter(X,Y,color='red')
plt.plot(X,reg.predict(X),color='blue')
plt.title('Support vector')
plt.xlabel('position')
plt.ylabel('Salary')
plt.show()


X_grid= np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,Y,color='red')
plt.plot(X_grid,reg.predict(X_grid),color='blue')
plt.title('Support vector')
plt.xlabel('position')
plt.ylabel('Salary')
plt.show()

