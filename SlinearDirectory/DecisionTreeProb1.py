

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing dataset

dataset= pd.read_csv("Position_Salaries.csv")

X= dataset.iloc[:,1:2].values
Y= dataset.iloc[:,2].values


#fitting dicisiontree regressor to dataset

from sklearn.tree import DecisionTreeRegressor

reg=DecisionTreeRegressor(random_state=0)

X=X.reshape(-1,1)
Y=Y.reshape(-1,1)
reg.fit(X,Y)

y_pred=reg.predict(X)


plt.scatter(X,Y, color='purple')

plt.plot(X,reg.predict(X),color='red')

plt.title('Decision tree regression')

plt.xlabel('Position')

plt.ylabel('Salary')

plt.show()


X_grid = np.arange(min(X),max(X),0.1)
X_grid= X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,color='purple')
plt.plot(X_grid,reg.predict(X_grid),color='red')
plt.title('Decision tree regression')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()