import numpy as np


def golden_section_search(f, x, direction, a, b, eps, counter, **kwargs):
    phi = (1 + 5 ** 0.5) / 2
    c = b - (b - a) / phi
    d = a + (b - a) / phi

    while abs(c - d) > eps:
        counter["f"] += 2
        if f(x + c * direction) < f(x + d * direction):
            b = d
        else:
            a = c
        c = b - (b - a) / phi
        d = a + (b - a) / phi

    return (b + a) / 2


def gradient_descent_with_golden_section(f, df, x0, counter, eps, max_iters=1000, **kwargs):
    result = []
    x = x0
    for _ in range(max_iters):
        result.append(x)
        counter["iter"] += 1
        grad = df(x)
        counter["df"] += 1
        step_size = golden_section_search(f, x, -grad, 0, 1, eps, counter, **kwargs)
        x_new = x - step_size * grad
        if np.linalg.norm(x_new - x) < eps:
            break
        x = x_new
    counter["x"] = len(result)
    return result
