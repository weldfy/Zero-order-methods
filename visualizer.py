import matplotlib
import numpy as np
import matplotlib.pyplot as plt


def visual_optimization3d(f, result, title, x_range, y_range, precision=100):
    x = np.linspace(x_range[0], x_range[1], precision)
    y = np.linspace(y_range[0], y_range[1], precision)
    X, Y = np.meshgrid(x, y)
    Z = f([X, Y])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
    for i, point in enumerate(result):
        if i != len(result) - 1:
            ax.scatter(point[0], point[1], f(point), color='red', label='Optimal Point')
    ax.scatter(result[len(result) - 1][0], result[len(result) - 1][1], f(result[len(result) - 1]), color='red',
               label='Optimal Point')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    ax.set_title(title)
    # plt.show()
    plt.savefig(f'3d/{title}.png')
    plt.close()


def visual_heat_lines2d(f, name, x_range, y_range, precision=100):
    x = np.linspace(x_range[0], x_range[1], precision)
    y = np.linspace(y_range[0], y_range[1], precision)
    X, Y = np.meshgrid(x, y)
    Z = f([X, Y])
    plt.contourf(X, Y, Z, cmap='RdGy')
    plt.colorbar()
    cs = plt.contour(X, Y, Z)
    plt.clabel(cs)
    # plt.show()
    plt.savefig(f'рисунок линий/heat lines {name}.png')
    plt.close()


def visual_trajectory(points, name):
    x = [x[0] for x in points]
    y = [x[1] for x in points]
    plt.plot(x, y, marker='o', linestyle='--', color='b')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'График траектории ({name})')
    # plt.show()
    plt.savefig(f'траектория/График траектории ({name}).png')
    plt.close()


def visual_minimizing(f, points, name):
    iters = list(range(len(points)))
    y = [f(x) for x in points]
    plt.plot(iters, y, marker='o', linestyle='--', color='b')
    plt.xlabel('Итерация')
    plt.ylabel('Значение целевой функции')
    plt.title(f'График минимизации ({name})')
    # plt.show()
    plt.savefig(f'минимизация/График минимизации ({name}).png')
    plt.close()
