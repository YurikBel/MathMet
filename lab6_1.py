import numpy as np
from numpy import linalg as la


def pvr(A, b):
    n = len(A)
    eps = 0.00001
    k = 0
    x1 = np.zeros((n, 1))
    x0 = np.zeros((n, 1))
    x1[0, 0] = x1[0, 0] + (
                (b[0, 0] - np.dot(A[0, 0:0], x1[0:0, 0]) - np.dot(A[0, 0 + 1:n], x1[0 + 1:n, 0])) / A[0, 0] - x1[0, 0])
    while np.abs(la.norm(x0 - x1)) > eps:
        x0 = np.copy(x1)
        for i in range(1, n):
            x1[i, 0] = x1[i, 0] +((b[i, 0] - np.dot(A[i, 0:i], x1[0:i, 0]) - np.dot(A[i, i + 1:n], x1[i + 1:n, 0])) / A[i, i] - x1[i, 0])
        k += 1
    return x1, k



def create():
    a = np.zeros((20, 20))
    for k in range(19):
        if k == 0:
            a[k][k] = 1
        else:
            a[k][k - 1] = 1
            a[k][k] = -2
            a[k][k + 1] = 1
    a[19][0] = 1
    for j in range(1, 19):
        a[19][j] = 2
    a[19][19] = 1

    b = np.zeros((20, 1))
    for i in range(20):
        if i == 0:
            b[i][0] = 1
        if i == 19:
            b[i][0] = - 20 / 3
        else:
            b[i][0] = 2 / (i + 1) ** 2
    return a, b


def Gauss(A, b):
    n = len(A)
    x = np.zeros((n, 1))
    for i in range(1, n):
        for j in range(i, n):
            g = A[j, i - 1]
            for k in range(n):
                A[j, k] = A[j, k] - A[i - 1, k] * g / A[i - 1, i - 1]
            b[j, 0] = b[j, 0] - b[i - 1, 0] * g / A[i - 1, i - 1]
    for i in range(n):
        c = 0
        for j in range(n):
            if (j != n - 1 - i):
                c += A[n - 1 - i, j] * x[j, 0]
        x[n - 1 - i, 0] = (b[n - 1 - i, 0] - c) / A[n - 1 - i, n - 1 - i]
    return x


a, b = create()
m = Gauss(a, b)
r = la.norm(b - a.dot(Gauss(a, b)))
a1 = la.inv(a)
nu = la.norm(a) * la.norm(a1)
print("Число обусловленности =", nu)
print('Значения х гаусс = ', m)
print("Норма вектора невязки = ", r)
x, k = pvr(a, b)
print('Значения х пвр= ', x)
print('Колво итераций= ', k)
w, v = la.eigh(a)
print('Минимальное собственное число', min(w))
print('Максимальное собственное число', max(w))
