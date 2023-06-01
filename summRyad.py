import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def f(x):
    return 36 / (x ** 2 + 5 * x + 4) #Заданная функция


def summa(n): # Функция считает сумму ряда
    summ = 0
    for i in range(n):
        summ += f(i)
    return summ


def ryadd(n): # Функция возвращает ряд частичных сумм
    lst = []
    for i in range(1, n):
        lst.append(summa(i))
    return lst


def limit(e): # Функция считает предел частичных сумм
    n = 0
    while summa(n + 1) - summa(n) >= e:
        n += 1
    return summa(n), n


def graphic(n): # Функция выводит графики
    x1 = range(n - 1)
    x2 = range(n)
    y1 = ryadd(n)
    y2 = []
    for i in range(n):
        y2.append(f(i))

    plt.plot(x1, y1, 'red')
    fig, ax = plt.subplots()
    xmin, xmax = ax.get_xlim()
    ax.set_xticks(np.arange(xmin, xmax, 5))
    plt.setp(ax.get_xticklabels(), **{'rotation': 45})
    ax.set_yticks(np.arange(xmin, 30, 5))
    plt.setp(ax.get_yticklabels(), **{'rotation': 45})

    plt.minorticks_on()
    plt.grid(which='major',
             color='k',
             linewidth=0.3)
    plt.grid(which='minor',
             color='k',
             linewidth=0.3
                )
    plt.xlabel('n')
    plt.ylabel('Sn')
    plt.show()

print(limit(0.00001))
print(ryadd(20))
graphic(100)

