from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Problem 02
C_0 = 1000
R = 8.314
n_A = 1

T = np.array([10,20,30,40,50,60,70]) + 273.15
K = np.array([0.060,0.12,0.22,0.40,0.69,0.98,1.3]) / C_0 ** n_A

X = np.array(1/T).reshape((-1, 1))
Y = np.log(K)

model = LinearRegression().fit(X,Y)

A = np.exp(model.intercept_)
E_A = model.coef_ * -R

print('Activation Energy:', E_A[0], ' J/mol')

x = np.arange(0,80,0.1) + 273.15
y = A * np.exp(-E_A / (R * x))

plt.plot(T,K,'o',color='black')
plt.plot(x,y)
plt.title('Problem 02')
plt.xlabel('Temperature (K)')
plt.ylabel('Rate Constant (m/s)')
plt.show()