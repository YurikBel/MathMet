import csv
import numpy as np
import matplotlib.pyplot as plt
import math


def read():
    with open('25_Pskov.csv', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=',')
        lst = []
        for row in reader:
            lst.append(row)
    return lst


def lagranz(x, y, t):
    z = 0
    for i in range(1, len(y)):
        p1 = 1
        p2 = 1
        for j in range(len(x)):
            if i != j:
                p1 = p1 * (t - x[j])
                p2 = p2 * (x[i] - x[j])
        z = z + y[i] * p1 / p2
    return z


def viewLagranz():
    lst = read()
    x = []
    y = []
    for i in range(13, 25, 1):
        x.append(float(lst[i][0]))
        y.append(float(lst[i][1]))
    xnew = np.linspace(np.min(x), np.max(x), 100)
    ynew = []
    for i in xnew:
        ynew.append(lagranz(x, y, i))
    plt.plot(x, y, 'o', xnew, ynew)
    plt.yticks(np.arange(min(ynew), max(ynew), 20))
    plt.grid(True)
    plt.show()


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


def newtonView():
    lst = read()
    x = []
    y = []
    for i in range(19, 25, 1):
        x.append(float(lst[i][0]))
        y.append(float(lst[i][1]))
    xnew = np.linspace(np.min(x), np.max(x), 100)
    ynew = []
    for i in xnew:
        ynew.append(newton(x, y, i))
    plt.plot(x, y, 'o', xnew, ynew)
    plt.yticks(np.arange(min(ynew), max(ynew), 1))
    plt.grid(True)
    plt.show()


def cof(y, m, k):
    if m == 1:
        return y[k + 1] - y[k]
    else:
        return con_diff(y, m - 1, k + 1) - con_diff(y, m-1, k)


def newton2(x, y, t):
    res = y[-1]
    h = 1
    q = (t - x[-1]) / h
    for i in range(1, len(y)):
        temp = 1
        for j in range(i):
            temp *= (q + j)
        res += temp * con_diff(y, i, len(y) - 1 - i) / math.factorial(i)
    return res


def newton2View():
    lst = read()
    x = []
    y = []
    for i in range(19, 25, 1):
        x.append(float(lst[i][0]))
        y.append(float(lst[i][1]))
    xnew = np.linspace(np.min(x), np.max(x), 100)
    ynew = []
    for i in xnew:
        ynew.append(newton2(x, y, i))
    plt.plot(x, y, 'o', xnew, ynew)
    plt.yticks(np.arange(min(ynew), max(ynew), 1))
    plt.grid(True)
    plt.show()


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



def approxView():
    lst = read()
    x = []
    y = []
    for i in range(1, len(lst), 1):
        if lst[i][1] == '999.9':
            continue
        x.append(float(lst[i][0]))
        y.append(float(lst[i][1]))
    xnew = np.linspace(np.min(x), np.max(x), 100)
    ynew = []
    for i in xnew:
        ynew.append(approx(x, y, i, 5))
    plt.plot(x, y, 'o', xnew, ynew)
    plt.yticks(np.arange(min(y), max(y), 1))
    plt.grid(True)
    plt.show()


def trigonom(x, y, t, n):
    x0 = []
    for i in range(len(y)):
        x0.append([])
        for j in range(2 * len(y)):
            h = 1
            if j == 0:
                x0[i].append(1)
            if j % 2 == 0:
                x0[i].append(np.cos(h * x[i]))
            else:
                x0[i].append(np.sin(h * x[i]))
                h += 1
    a = (np.array(x0).transpose().dot(x0)).dot(np.array(x0).transpose()).dot(y)
    approx_function = 0
    for i in range(len(a / 2)):
            approx_function += (a[0] * np.cos(i * t) + a[0] * np.sin(i * t))
    return approx_function

def trigonomView():
    lst = read()
    x = []
    y = []
    for i in range(1, len(lst), 1):
        if lst[i][1] == '999.9':
            continue
        x.append(float(lst[i][0]))
        y.append(float(lst[i][1]))
    xnew = np.linspace(np.min(x), np.max(x), 100)
    ynew = []
    for i in xnew:
        if trigonom(x, y, i, 4) < -20 or trigonom(x, y, i, 4) > 0 :
            ynew.append(1)
        else:
            ynew.append(trigonom(x, y, i, 4))
    plt.plot(x, y, 'o', xnew, ynew)
    plt.yticks(np.arange(min(y), max(y), 1))
    plt.grid(True)
    plt.show()

trigonomView()