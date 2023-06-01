from sympy import Symbol, diff, exp, cos, sin, Matrix, var, symbols
import sympy as sym
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy import interpolate
from numpy.linalg import norm

n = 2  # количество переменных
y00 = np.array([1, 0])  # начальные условия y(0)=1,y'(0)=0
Tn0, tmax0 = 0, 2  # временной интервал
h = 0.01  # начальный шаг
eps = 0.01  # точность


t=Symbol('t')
y = sym.Matrix(n, 1, lambda i,j:var('y[%d]' % (i)))

def f(T, y):
    return Matrix([ \
        y[1], \
        -4 * y[0] + T * sin(T), \
        ])


def pfpt(t,y):
    return np.array(
    str(np.array2string(np.array(f(t, y).diff(t)).T[0], separator=', ')).replace('Matrix(', ''))