import numpy as np
import matplotlib.pyplot as plt


x=np.array([1,2,3,4]) #This is the input features
y= np.array([1,2,3,4]) #This is the training set 
m=x.shape[0] #Stores the length of x in m

def compute_cost(x,y,w,b): #funciton to compute cost
    sum=0
    m=x.shape[0]
    for i in range(m):
        f=w*x[i]+b #calculate predicted value f 
        sum=sum + ( f-y[i])**2 
    j=(1/(2*m))*sum #j is the cost
    print("cost:", j)
    return j

def compute_gradient(x,y,w,b):
    m=x.shape[0]
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):
        f=w*x[i]+b
        dj_db_i=f-y[i]
        dj_dw_i=(f- y[i] )*x[i]
        dj_db += dj_db_i 
        dj_dw= dj_dw_i
    dj_db=dj_db/m
    dj_dw=dj_dw/m
    print(dj_db, dj_dw)
    return dj_dw, dj_db

    








w, b=8, 3

compute_cost(x,y,w,b)
compute_gradient(x,y,w,b)






plt.plot(x,y,marker="x", label="try") #plot of x vs y
plt.legend()
plt.show()

