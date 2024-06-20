import numpy as np
import matplotlib.pyplot as plt


x=np.array([1,2,3,4]) #This is the input features
y= np.array([1,2,3,4]) #This is the training set or output variable
m=x.shape[0] #Stores the length of x in m

f=np.zeros(m) #creates a numpy array and initializes it with zeroes
for i in range (m):
    f[i]=w*x[i]+b #linear regression model, f is the model's prediction
sum=0


for i in range(m):
    sum=sum + ( f[i]-y[i])**2 
j=(1/(2*m))*sum #j is the cost
print(j)

plt.plot(x,y,marker="x", label="try") #plot of x vs y
plt.plot(x,f,marker="o", c= "r") #plot of the linear regression model
plt.legend()
plt.show() 

