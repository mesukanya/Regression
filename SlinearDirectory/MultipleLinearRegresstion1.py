import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv("/home/admin1/Downloads/50_Startups.csv")

#IMporting the dataset

X= data.iloc[:,:-1].values
Y= data.iloc[:, 4].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_encoder_x= LabelEncoder()
X[:,3] = label_encoder_x.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()


#avoiding the dummy variable

X=X[:,1:]


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
reg = LinearRegression()

x_train= x_train.reshape(-1,1)
y_train=y_train.reshape(-1,1)
x_test=x_test.reshape(-1,1)
reg.fit(x_train,y_train)
y_pred= reg.predict(x_test)

print(y_pred)

#States=pd.get_dummies(X['State'],drop_first=True)

#X=X.drop('State',axis=1)

#X= pd.concat([X,States],axis=1)


#plt.scatter(X,Y, color='r')



#m = len(X)
#data_count = X.reshape(m,1)
#data_humidity = data_humidity.reshape(m,1)

#y_pred = reg.predict(x_train)
#plt.plot(x_train, y_pred, color='b')
#r2 = reg.score(x_train, y_train)
#print(r2)
#plt.show()