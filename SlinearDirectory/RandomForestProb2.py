import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('bike_sharing.csv')

X=dataset.iloc[:,10].values
Y=dataset.iloc[:,16].values

X=X.reshape(-1,1)


from sklearn.ensemble import RandomForestRegressor
reg=RandomForestRegressor(n_estimators=10,random_state=0)
reg.fit(X,Y)

y_predict=reg.predict(X)

X_grid= np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)

plt.scatter(X,Y,color='red')
plt.plot(X_grid,reg.predict(X_grid),color='blue')
plt.title('RandomForest')
plt.xlabel('temp')
plt.ylabel('Count')
plt.show()

