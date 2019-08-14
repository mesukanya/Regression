import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset=pd.read_csv("/home/admin1/Desktop/SlinearRegresstion/SlinearDirectory/Position_Salaries.csv")

X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:,2].values

from sklearn.ensemble import RandomForestRegressor
reg=RandomForestRegressor(n_estimators=10,random_state=0)
reg.fit(X,Y)


y_predict= reg.predict(X)


X_grid=np.arange(min(X),max(X),0.1)
print("arrange",X_grid)
X_grid=X_grid.reshape(len(X_grid),1)
print("reshape",X_grid)
plt.scatter(X,Y,color='red')
plt.plot(X_grid,reg.predict(X_grid),color='blue')

plt.title('RandomForest')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()










