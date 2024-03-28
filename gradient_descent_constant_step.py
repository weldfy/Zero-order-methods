import numpy as np


def gradient_descent_constant_step(f, df, x0, eps, counter, learning_rate=0.1, max_iters=1000, **kwargs):
    result = []
    x = x0
    for i in range(max_iters):
        result.append(x)
        counter["iter"] += 1
        counter["df"] += 1
        grad = df(x)
        x_new = x - learning_rate * grad
        if np.linalg.norm(x_new - x) < eps:
            break
        x = x_new
    counter["x"] = len(result)
    return result