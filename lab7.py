import numpy as np
import matplotlib.pyplot as plt

def func(x1, x2):
    return 2 * x1 ** 2 + 4 * x1 + 3 * (x2 ** 2) - 39.0 * x2 + 129.75

x_0 = np.array([0, -3])
epsilon = 0.001


def difx(x1, x2):
    return 4 * x1 + 4


def dify(x1, x2):
    return 6 * x2 - 39.0


def Newton(x_0, epsilon):
    iters = 0
    x_new = [x_0[0]]
    y_new = [x_0[1]]

    x_1 = np.copy(x_0) + 1
    while abs(np.linalg.norm(x_0 - x_1)) and abs(func(x_0[0], x_0[1]) - func(x_1[0], x_1[1])) > epsilon:
        x_0 = np.copy(x_1)
        gessian = np.array(
            [[4, 0], [0, 6]])
        f_diff = np.array([difx(x_0[0], x_0[1]), dify(x_0[0], x_0[1])])
        x_1 = x_0 - np.dot(np.linalg.inv(gessian), f_diff)
        x_new.append(x_1[0])
        y_new.append(x_1[1])
        iters += 1

    return x_1, iters, x_new, y_new


result = Newton(x_0, epsilon)
print("минимум: ", result[0])

fig = plt.figure(figsize=(1,1))
ax = fig.add_subplot(111, projection='3d')
x_1 = np.arange(-3, 3, 0.1)
x_2 = np.arange(4, 9, 0.1)
x_1_grid, x_2_grid = np.meshgrid(x_1, x_2)
ax.plot_surface(x_1_grid, x_2_grid, func(x_1_grid, x_2_grid))
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.show()


xlist = np.linspace(-3, 3, 1000)
ylist = np.linspace(-4, 9, 1000)
X, Y = np.meshgrid(xlist, ylist)
Z = 2 * X ** 2 + 4 * X + 3 * (Y ** 2) - 39.0 * Y + 129.75
fig, ax = plt.subplots(1, 1)
cp = ax.contour(X, Y, Z, levels=100)
fig.colorbar(cp)  # Add a colorbar to a plot
ax.scatter(result[2], result[3], color="red")
ax.plot(result[2], result[3], color="red")
plt.show()
