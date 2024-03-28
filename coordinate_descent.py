import numpy as np

import gradient_descent_with_golden_section


def minarg_with_golden_section(f, x, i, eps, counter):
    direction = np.asarray([0 if j != i else 1 for j in range(len(x))])
    alpha = gradient_descent_with_golden_section.golden_section_search(f, x, direction, -1, 1, eps, counter)
    return x[i] + alpha


def coordinate_descent(f, x0, eps, counter, max_iters=1000, **kwargs):
    result = []
    x = x0
    for i in range(max_iters):
        result.append(x)
        counter["iter"] += 1
        x_new = np.asarray([xi for xi in x])
        for i in range(len(x0)):
            x_new[i] = minarg_with_golden_section(f, x, i, eps, counter)
        counter["f"] += 2
        if abs(f(x_new) - f(x)) < eps:
            break
        x = x_new
    counter["x"] = len(result)
    return result
