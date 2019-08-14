import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import  DecisionTreeClassifier
import pickle
from sklearn.svm import SVR


data = pd.read_csv('/home/admin1/Desktop/SlinearRegresstion/Classification/DecisionTree/Social_Network_Ads.csv')
X= data.iloc[:,2:3].values
y=data.iloc[:,3].values
print("X",X)
print('y',y)

print(len(data))


X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.25,random_state=0)
print(len(X_test),len(X_train))


sc = StandardScaler()

X_train = sc.fit_transform(X_train.reshape(-1,1))
X_test=sc.transform(X_test)


Classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
Classifier.fit(X_train,y_train)
y_predict= Classifier.predict(X_test)
print("predict",y_predict)
print("y test",y_test)


#importing csv file

dataset=pd.read_csv('Social_Network_Ads.csv')
length_old = len(dataset.columns)

#handling categorical data

gender=pd.get_dummies(dataset["Gender"])
dataSet = dataset.drop('Gender', axis=1)
dataSet = pd.concat([dataSet, gender], axis=1)


#Spliting dataset into two csv files

dt_training= dataset.sample(frac=0.7)
dt_testing=df_test = pd.concat([dataset, dt_training]).drop_duplicates(keep=False)
length_new = len(dataset.columns)
y_index = dataSet.columns.get_loc("EstimatedSalary")
print(y_index)
dt_training.to_csv('training_data.csv', header=True, index=None)
dt_testing.to_csv('test_data.csv', header=True, index=None)

dataSet = pd.read_csv('training_data.csv')

#save model
file_name = '/DecisionTree.pkl'
pkl_file = open(file_name, 'wb')
print("pickelfile",pkl_file)
model = pickle.dump(Classifier, pkl_file)
print("final model",model)


#draw plot

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)

# Visualising the Training set Resilts
from matplotlib.colors import ListedColormap
x_set, y_set = X_train, y_train
x1,x2 = np.meshgrid(np.arange(start = x_set[:, 0].min()-1, stop = x_set[:, 0].max()+1, step = 0.01),
                np.arange(start = x_set[:, 1].min()-1,stop = x_set[:, 1].max()+1, step = 0.01))
plt.contourf(x1,x2, Classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('Decision Tree (Training Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()











