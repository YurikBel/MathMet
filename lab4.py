import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl


content = np.fromfile("var25_z1.bin",  dtype=np.int16, count=20000)
x = [i * 0.001 for i in range(len(content))]
N = 10
v = 1000
temp = N * v
middle = np.zeros(len(content), float)

def sinWindow(n):
    return np.sin((np.pi * n) / 10000)


def average(arr, temp, mid):
    summ = 0
    for i in range(temp + 1):
        summ += sinWindow(i)
    for i in range(temp, len(arr)):
        k = 0
        for j in range(i - temp, i + 1):
            k += arr[j] * sinWindow(temp - (i - j))
        h = int((2 * i - temp) / 2)
        mid[h] = k / (summ)
    return mid


ampl = average(content, temp, middle)
# amplArr = [x for x in ampl]
# left = amplArr.index(5)
# right = amplArr.index(15)
pl.figure(1)
plt.plot(x, content)
plt.grid()

pl.figure(2)
plt.plot(x, ampl)
plt.grid()

pl.figure(3)
plt.plot(x, ampl)
plt.ylim(210, 240)
plt.grid()

plt.show()
#
# content = np.fromfile("var25_z2.bin", dtype=np.double)
#
# spectr = []
# odd = []
# even = []
# chastots = []
# v = 1000
# N = len(content)
#
#
# def sinWindow(n):
#     return np.sin((np.pi * n) / 10000)
#
#
# def dpf(k, n):
#     return np.exp(complex(0, (-2 * np.pi * k) / n))
#
# def differ(odd, even):
#     for i in range(len(content)):
#         if i % 2 == 0:
#             even.append(content[i])
#         else:
#             odd.append(content[i])
#     return odd, even
#
#
# def s(arr, k):
#     c = 0
#     for m in range(int(N / 2)):
#         c += arr[m] * sinWindow(m) * dpf(k * m, N / 2)
#     return c
#
#
# def ryad(odd, even, chastots, spectr):
#     for k in range(int(N / 2)):
#         s0 = s(even, k)
#         s1 = s(odd, k)
#         spectr.append(abs(s0 + dpf(k, N) * s1) / N)
#         if ((abs(s0 + dpf(k, N) * s1) / N) > 1 / 100000):
#             chastots.append(len(spectr) - 1)
#         spectr.append(abs(s0 - dpf(k, N) * s1) / N)
#         if ((abs(s0 - dpf(k, N) * s1) / N) > 1 / 100000):
#             chastots.append(len(spectr) - 1)
#     return chastots, spectr
#
#
# x = [i * 0.001 for i in range(len(content))]
# x2 = np.arange(0, len(content), 1)
#
# pl.figure(1)
# plt.plot(x, content)
# plt.grid()
#
# odd, even = differ(odd, even)
# chastots, spectr = ryad(odd, even, chastots, spectr)
# pl.figure(2)
# plt.plot(x2, spectr)
# plt.yscale('log')
# plt.grid()


# content = np.fromfile("var25_z2.bin", dtype=np.double)
# w = np.fft.fft(content)
# m = np.max(np.abs(w))
# neww = [20*np.log10(np.abs(i)/m) for i in w]
# freqs = np.fft.fftfreq(len(content), d=1/1000)
#
# pl.figure(3)
# plt.plot(freqs, np.array(neww))
# plt.grid()
#
# plt.show()

