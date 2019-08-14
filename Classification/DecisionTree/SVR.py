import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap


dataset=pd.read_csv('/home/admin1/Desktop/SlinearRegresstion/Classification/SVR/Social_Network_Ads.csv')
X=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,4].values
print(y)
print(X)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


classifier=SVC(kernel='linear')
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)


# Visualising the Training set Results

x_set, y_set = X_train, y_train
x1,x2 = np.meshgrid(np.arange(start = x_set[:, 0].min()-1, stop = x_set[:, 0].max()+1, step = 0.01),
                    np.arange(start = x_set[:, 1].min()-1, stop = x_set[:, 1].max()+1, step = 0.01))
plt.contourf(x1,x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('Logistic Regression (Training Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Testing set Results

x_set, y_set = X_test, y_test
x1,x2 = np.meshgrid(np.arange(start = x_set[:, 0].min()-1, stop = x_set[:, 0].max()+1, step = 0.01),
                    np.arange(start = x_set[:, 1].min()-1, stop = x_set[:, 1].max()+1, step = 0.01))
plt.contourf(x1,x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('Logistic Regression (Testing Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


#importing csv
dataset=pd.read_csv('/home/admin1/Desktop/SlinearRegresstion/Classification/SVR/Social_Network_Ads.csv')

#heading categorical data
gender=pd.get_dummies(dataset['Gender'])
dataset=dataset.drop('Gender',axis=1)
dataset=pd.concat([dataset,gender],axis=1)

#spliting data into two csv files
dt_training=dataset.sample(frac=0.7)
dt_testing=pd.concat([dataset,dt_training]).drop_duplicates(keep=False)
dt_training.to_csv('training_data.csv', header=True, index=None)
dt_testing.to_csv('test_data.csv', header=True, index=None)


#Save model
File_name='SVR.pkl'
pkl_file=open(File_name,'wb')
model=pickle.dump(classifier,pkl_file)
