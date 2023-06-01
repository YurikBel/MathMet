import matplotlib.pyplot as plt
import numpy as np
v = np.linspace(0, 7, 100)
x = v * 10**(14)
c = 3 * 10**8
h = 6.6 * 10**(-34)
k = 1.38 * 10**(-23)
t = 3000
u = 2 * 3.14 * h * (x ** 3) * (1 / c**2) * (1 / (np.exp((h * x) / (k * t)) - 1))

plt.plot(v, u)
plt.minorticks_on()
plt.grid(which='major',
         color='k',
         linewidth=0.5)
plt.grid(which='minor',
         color='k',
         linestyle=':')
plt.show()