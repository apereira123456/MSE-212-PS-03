from sklearn.linear_model import LinearRegression
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

C_0 = 1000
R = 8.314

# Problem 01 Part E
K_1 = 2.31 / C_0

C_A = C_0 / np.exp(K_1 * 10 * 60)

K_2 = 2.31 / (C_A)

T_1 = 27 + 273.15
T_2 = 40 + 273.15

def f(variables):
    (A,E_A) = variables
    Eq_1 = K_1 - A * np.exp(-E_A / (R * T_1))
    Eq_2 = K_2 - A * np.exp(-E_A / (R * T_2))
    return [Eq_1,Eq_2]

solution_1 = fsolve(f, (0.1,1))
print('Activation Energy:', solution_1[1], ' J/mol')

# Problem 02
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

# Problem 3 Part E
r, q_A, q_B = sp.symbols('r, q_A, q_B')
k, k_A, C_A, k_B, C_B = sp.symbols('k, k_A, C_A, k_B, C_B')
eqns = [k_A * C_A * (1 - q_A - q_B) - 1 / k_A * q_A - r, \
        k_B * C_B * (1 - q_A - q_B) - 2 * r, \
        k * q_A * q_B - r]
sol = sp.nonlinsolve(eqns, r, q_A, q_B)

sp.pprint(sol.args[1], use_unicode=True)