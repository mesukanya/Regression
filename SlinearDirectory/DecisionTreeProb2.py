

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv('/home/admin1/Desktop/SlinearRegresstion/SlinearDirectory/bike_sharing.csv')

X= dataset.iloc[:,10].values
Y=dataset.iloc[:,16].values

X=X.reshape(-1,1)

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=17300,random_state=0)


from sklearn.tree import DecisionTreeRegressor

reg= DecisionTreeRegressor(random_state=0)

reg.fit(X_train,Y_train)

y_predict=reg.predict(X_test)

X_grid = np.arange(min(X_train),max(X_train),0.001)
X_grid= X_grid.reshape((len(X_grid),1))
plt.scatter(X_train,Y_train,color='purple')
plt.plot(X_grid,reg.predict(X_grid),color='red')
plt.title('Decision tree regression')
plt.xlabel('Tem')
plt.ylabel('Count')
plt.show()








