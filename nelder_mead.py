from scipy.optimize import minimize
import numpy as np

import functions


def nelder_mead(f, x0, bounds, eps, counter, max_iters=1000, **kwargs):
    res = minimize(f, np.asarray(x0), bounds=bounds, method='Nelder-Mead', tol=eps,
                   options={"maxiter": max_iters, "return_all": True})
    counter["iter"] = res.nit
    counter["f"] = "?"
    counter["df"] = 0
    counter["x"] = len(res.allvecs)
    return res.allvecs