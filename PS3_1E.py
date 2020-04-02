from scipy.optimize import fsolve
import numpy as np

# Problem 01 Part E
C_0 = 1000
R = 8.314

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