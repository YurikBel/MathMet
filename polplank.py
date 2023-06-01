import csv
import numpy as np
import matplotlib.pyplot as plt
import math


def con_diff(y, m, k):
    if m == 1:
        return y[k + 1] - y[k]
    else:
        return con_diff(y, m - 1, k + 1) - con_diff(y, m-1, k)


def newton(x, y, t):
    res = y[0]
    h = 1
    q = (t - x[0]) / h
    for i in range(1, len(y)):
        temp = 1
        for j in range(i):
            temp *= (q - j)
        res += temp * con_diff(y, i, 0) / math.factorial(i)
    return res


def approx(x, y, t, n):
    x0 = []
    for i in range(len(x)):
        x0.append([])
        for j in range(n + 1):
            x0[i].append(x[i] ** j)
    a = (np.linalg.inv(np.array(x0).transpose().dot(x0))).dot(np.array(x0).transpose()).dot(y)
    approx_function = a[0]
    for i in range(1, len(a)):
        approx_function += a[i] * t ** i
    return approx_function

def newtonView():
    with open(f'zad2pogr.dat') as f:
        lines = f.read().splitlines()
        x_name, y_name, x_err, y_err, log, err = [k.strip() for k in lines[0].split(',')]
        x_value = []
        y_value = []
        x_value_err = []
        y_value_err = []
        for line in lines[1:]:
            val = line.split()
            x, y, x_e, y_e = val
            x_value.append(float(x))
            y_value.append(float(y))
            x_value_err.append(x_e)
            y_value_err.append(y_e)
    xnew = np.linspace(np.min(x_value), np.max(x_value), 100)
    ynew = []
    for i in xnew:
        ynew.append(approx(x_value, y_value, i, 3))
    y2 = (100) / (xnew + 0.09)
    ax = plt.subplot()
    ax.set_xticks(np.arange(0, 3.5, 0.05))
    plt.setp(ax.get_xticklabels(), **{'rotation': 45})
    ax.set_yticks(np.arange(0, 50, 2))
    plt.setp(ax.get_yticklabels(), **{'rotation': 45})
    plt.plot(x_value, y_value, 'o', xnew, ynew)
    plt.plot(xnew, y2)
    plt.grid(True)
    plt.minorticks_on()
    plt.grid(which='major',
             color='k',
             linewidth=0.5)
    plt.grid(which='minor',
             color='k',
             linewidth=0.3
             )
    plt.show()

newtonView()