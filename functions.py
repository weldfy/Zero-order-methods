import random

import numpy as np

def quadratic_factory_f(cf):
    return lambda x: (x[0] ** 2) * cf[0] + (x[1] ** 2) * cf[1] + (x[0] * x[1]) * cf[2] + x[0] * cf[3] + x[1] * cf[4]


def quadratic_factory_df(cf):
    return lambda x: np.asarray([2. * x[0] * cf[0] + x[1] * cf[2] + cf[3], 2. * x[1] * cf[1] + x[0] * cf[2] + cf[4]])


quadratic = [
    {
        "x_range": [-10., 10.],
        "y_range": [-10., 10.],
        "f": quadratic_factory_f([0.6, 0.7, 0.12, 0, 0]),
        "df": quadratic_factory_df([0.6, 0.7, 0.12, 0, 0])
    },
    {
        "x_range": [-10., 10.],
        "y_range": [-10., 10.],
        "f": quadratic_factory_f([2, 2, 2, 6, 0]),
        "df": quadratic_factory_df([2, 2, 2, 6, 0])
    },
]


def get_point(a, b):
    t = random.random()
    a, b = np.asarray([a[0], b[0]]), np.asarray([a[1], b[1]])
    return a + t*(b-a)