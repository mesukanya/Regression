import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import pickle

data = pd.read_csv('Position_Salaries.csv')
X = data.iloc[:, 1:2].values
y = data.iloc[:, 2].values
length_old = len(data.columns)
print(X)
print(y)
#print(length_old)

sc_X = StandardScaler()
sc_y = StandardScaler()
print("sc_x",sc_X)
print("sc_y",sc_y)
X = sc_X.fit_transform(X.reshape(-1,1))
y = sc_y.fit_transform(y.reshape(-1,1))

print("Print x",X)
print("Print y",y)

reg = SVR(kernel='rbf')
print("reg",reg)

reg.fit(X,y)
print("after fit",reg)

#y_pred = reg.predict(np.array([5]).reshape(-1,1))

r2 = reg.score(X,y)
print(r2)

#y_pred = reg.predict(np.array([6.5]).reshape(-1,1))
#print(y_pred)
plt.scatter(X,y, color='r')
plt.plot(X, reg.predict(X),color='b')
plt.show()


# Importing dataset
dataSet = pd.read_csv('Position_Salaries.csv')
length_old = len(dataSet.columns)

# Handling categorical data
positions = pd.get_dummies(dataSet['Position'])
print(positions)
dataSet = dataSet.drop('Position', axis=1)
print(dataSet)
dataSet = pd.concat([dataSet, positions], axis=1)
print(dataSet)

# Splitting dataset into 2 different csv files
df_training = dataSet.sample(frac=0.7)
print("Training data",df_training)
df_test = pd.concat([dataSet, df_training]).drop_duplicates(keep=False)
print("Testing data",df_test)
length_new = len(dataSet.columns)
y_index = dataSet.columns.get_loc("Salary")
print(y_index)
df_training.to_csv('training_data.csv', header=True, index=None)
df_test.to_csv('test_data.csv', header=True, index=None)

dataSet = pd.read_csv('training_data.csv')

#save model
file_name = 'RandomForestRegression.pkl'
pkl_file = open(file_name, 'wb')

model = pickle.dump(reg, pkl_file)

print("model",model)
# Loading pickle model to predict data from test_data.csv file
pkl_file = open(file_name, 'rb')
model_pkl = pickle.load(pkl_file)

dataSet_testdata = pd.read_csv('test_data.csv')

x_testdata = dataSet_testdata.iloc[:, (len(data.columns)-1): len(dataSet)]
y_testdata = dataSet_testdata.iloc[:, y_index:(y_index+1)]
y_pred_pkl = model_pkl.predict(np.array([6.5]).reshape(-1,1))

print(y_pred_pkl)