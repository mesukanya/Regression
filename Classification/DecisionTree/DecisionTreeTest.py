import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score


file_name = '/home/admin1/Desktop/SlinearRegresstion/Classification/DecisionTree/DecisionTree.pkl'

# Loading pickle model to predict data from test_data.csv file
pkl_file = open(file_name, 'rb')
model_pkl = pickle.load(pkl_file)

dataSet_testdata = pd.read_csv('/home/admin1/Desktop/SlinearRegresstion/Classification/DecisionTree/test_data.csv')

x1_index = dataSet_testdata.columns.get_loc("Age")
y_index = dataSet_testdata.columns.get_loc("EstimatedSalary")

print(x1_index)
print(y_index)
x_testdata = dataSet_testdata.iloc[:, [x1_index]]
#y_testdata = dataSet_testdata.iloc[:, y_index:(y_index+1)]
y_testdata = dataSet_testdata.iloc[:, [y_index]]
print("x data",x_testdata)
print("y data",y_testdata)

y_pred = model_pkl.predict(x_testdata)

print(y_pred)