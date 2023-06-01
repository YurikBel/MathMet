import numpy as np
from numpy import linalg as la


def Gauss(A, b):
    n = len(A)
    x = np.zeros((n, 1))
    # треугольный вид

    for i in range(1, n):
        for j in range(i, n):
            g = A[j, i - 1]
            for k in range(n):
                A[j, k] = A[j, k] - A[i - 1, k] * g / A[i - 1, i - 1]
            b[j, 0] = b[j, 0] - b[i - 1, 0] * g / A[i - 1, i - 1]

    # поиск решений x
    for i in range(n):
        c = 0
        for j in range(n):
            if (j != n - 1 - i):
                c += A[n - 1 - i, j] * x[j, 0]
        x[n - 1 - i, 0] = (b[n - 1 - i, 0] - c) / A[n - 1 - i, n - 1 - i]
    return x
# матрица левой части
a = np.zeros((20, 20))
i = 0
for k in range(19):
    if k == 0:
            a[k][k] = 1
    else:
        a[k][k-1] = 1
        a[k][k] = -2
        a[k][k+1] = 1


a[19][0] = 1
for j in range(1, 19):
    a[19][j] = 2
a[19][19] = 1
# матрица правой части
b = np.zeros((20, 1))
for i in range(20):
    if i == 0:
        b[i][0] = 1
    if i == 19:
        b[i][0] = - 20 / 3
    else:
        b[i][0] = 2 / (i + 1)**2
x = Gauss(a, b)
r = la.norm(b - a.dot(Gauss(a, b)))
a1 = la.inv(a)
nu = la.norm(a) * la.norm(a1)
print("число обусловленности =", nu)
print('значения х = ', x)
print("норма вектора невязки = ", r)



