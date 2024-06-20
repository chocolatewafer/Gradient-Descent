import numpy as np
import matplotlib.pyplot as plt


x=np.array([1,2,3,4])
y= np.array([1,2,3,4])
m=x.shape[0]
print(m)

f=np.zeros(m)
for i in range (m):
    f[i]=w*x[i]+b
sum=0
for i in range(m):
    sum=sum + ( f[i]-y[i])**2
j=(1/(2*m))*sum
print(j)

plt.plot(x,y,marker="x", label="try")
plt.plot(x,f,marker="o", c= "r")
plt.legend()
plt.show()

