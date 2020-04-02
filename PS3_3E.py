import sympy as sp

# Problem 3 Part E
r, q_A, q_B = sp.symbols('r, q_A, q_B')
k, k_A, C_A, k_B, C_B = sp.symbols('k, k_A, C_A, k_B, C_B')
eqns = [k_A * C_A * (1 - q_A - q_B) - 1 / k_A * q_A - r, \
        k_B * C_B * (1 - q_A - q_B) - 2 * r, \
        k * q_A * q_B - r]
sol = sp.nonlinsolve(eqns, r, q_A, q_B)

sp.pprint(sol.args[1], use_unicode=True)

sp.latex(eval(sol))