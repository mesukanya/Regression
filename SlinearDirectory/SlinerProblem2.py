import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#collecting x and y value

data= pd.read_csv('/home/admin1/Downloads/bike_sharing.csv')

Y= data['cnt'].values
X= data['temp'].values

print(X)
print(Y)

mean_x= np.mean(X)
mean_y=np.mean(Y)

#total no of values
m= len(X)

#using formula calculate b1 and b0

number=0
denom=0

for i in range(m):
    number +=(X[i]-mean_x)*(Y[i]-mean_y)
    denom +=(X[i]-mean_x)**2

    b1=number/denom
    b0=mean_y-(b1*mean_x)

    #print coefficients

print(b1,b0)

#plotting values and regresstion line

max_x= np.max(X)
min_x =np.min(X)

#calculating  line values x and y

x = np.linspace(min_x,max_x,50)
y= b0 + b1 * x


#Plotting line

plt.plot(x,y , color='red',label='Regresstion Line')
plt.scatter(X,Y,color='blue',label='Scatter plot')
plt.xlabel('Temperature')
plt.ylabel('Count')
plt.legend()
plt.show()

