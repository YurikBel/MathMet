
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy import interpolate
from numpy.linalg import norm

n = 2  # количество переменных
y00 = np.array([1, 0])  # начальные условия y(0)=1,y'(0)=0
xmin, xmax = 0, 2  # временной интервал
h = 0.01
eps = 0.01  # точность
y = [0, 0]

def f(x, y):
    return np.array([y[1], -4 * y[0] + x * np.sin(x)])


def dfdt(x, y):
    return [0, x * np.cos(x) + np.sin(x)]


def d2fdt2(x, y):
    return [0, -x * np.sin(x) + 2 * np.cos(x)]


def J(x, y):
    return [[0, 1], [-4, 0]]


def dJdt(x, y):
    return [[0, 0], [0, 0]]


def model(x, y):
    return [y[1], x * np.sin(x) - 4 * y[0]]


Time = np.array([])
Y = np.empty((0, n), np.float32)

N = 0
while True:
    h = h / 2

    yn = y00
    Tn = xmin
    tmax = xmax

    N += 1

    OldTime = Time
    OldY = Y

    Time = np.array([])
    Y = np.empty((0, n), np.float32)

    while Tn <= tmax:
        Tn = Tn + h
        yo = yn
        Time = np.append(Time, Tn)

        Y = np.append(Y, [yo], axis=0)

        const_pfpt = dfdt(Tn, yn)
        const_J = J(Tn, yn)
        const_F = model(Tn, yn)
        yn = np.array(
            yn + h * (const_F + h / 2 * (const_pfpt + np.dot(const_J, const_F))
                      + h ** 2 / 6 * (d2fdt2(Tn, yn) +
                        np.dot(dJdt(Tn, yn),
                               const_F) + np.dot(
                        const_J, const_pfpt))))
    if N >= 3:
        YY = Y
        YO = OldY
        if len(YY) % 2 != 0:
            YY = YY[:-1]
        if len(YO) % 2 != 0:
            YO = YO[:-1]
        YY = np.array(YY, dtype=np.float32)
        YO = np.array(YO, dtype=np.float32)
        delta = np.max(norm(YY[0::2] - YO, axis=0))
        if delta < eps:
            break

solve = solve_ivp(model, [xmin, xmax], y00, method='LSODA', min_step=0.01, max_step=0.01)

Time2 = solve.t
Z = solve.y

Z1, Z2 = Z
Y1, Y2 = Y.T

f, (ax1, ax2, ax3) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [3, 3, 3]})
f.set_size_inches(11, 3)

ax1.set_title('Графики решений')
ax1.plot(Time, Y1, '*', markersize=1)
ax1.plot(Time2, Z1)
ax1.set_ylabel("y")
ax1.set_xlabel("t")

ax2.plot(Time, Y2, '*', markersize=1)
ax2.plot(Time2, Z2)
ax2.set_ylabel("y'")
ax2.set_xlabel("t")

ax3.plot(Y1, Y2, '*', markersize=1)
ax3.plot(Z1, Z2)
ax3.set_xlabel("y")
ax3.set_ylabel("y'")

f.tight_layout()
plt.show()

x = np.linspace(np.min(Time), np.max(Time), 1000)
f = interpolate.interp1d(Time, Y1, axis=0, fill_value="extrapolate")
Y1f = f(x)
f = interpolate.interp1d(Time2, Z1, axis=0, fill_value="extrapolate")
Z1f = f(x)

x = np.linspace(np.min(Time), np.max(Time), 1000)
f = interpolate.interp1d(Time, Y2, axis=0, fill_value="extrapolate")
Y2f = f(x)
f = interpolate.interp1d(Time2, Z2, axis=0, fill_value="extrapolate")
Z2f = f(x)

f, (plt1, plt2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1]})
f.set_size_inches(11, 3)

plt1.set_title('Разностные графики')
plt1.plot(x, Z1f - Y1f)
plt1.set_ylabel("y'")
plt1.set_xlabel("t")

plt2.plot(x, Z2f - Y2f)
plt2.set_ylabel("y")
plt2.set_xlabel("t")

f.tight_layout()
plt.show()
