from matplotlib import pyplot as plt

import coordinate_descent
import nelder_mead
import gradient_descent_constant_step
import gradient_descent_with_golden_section
import functions
import visualizer

tasks = [
    {
        "name": "Градиентный спуск с постоянным шагом",
        "method": gradient_descent_constant_step.gradient_descent_constant_step
    },
    {
        "name": "Градиентный спуск с золотым сечением",
        "method": gradient_descent_with_golden_section.gradient_descent_with_golden_section
    },
    {
        "name": "Метод Нелдера-Мида",
        "method": nelder_mead.nelder_mead
    },
    {
        "name": "Метод покоординатного спуска",
        "method": coordinate_descent.coordinate_descent
    }
]


def show_task(task, f, x0):
    x_range = f["x_range"]
    y_range = f["y_range"]
    cnt = {
        "f": 0,
        "df": 0,
        "eps": float(1e-6),
        "iter": 0,
        "x": 0,
        "max_iters": 1000
    }
    bounds = [x_range, y_range]
    res = task["method"](f=f["f"], df=f["df"], x0=x0, bounds=bounds, eps=cnt["eps"], counter=cnt,
                         max_iters=cnt["max_iters"])
    visualizer.visual_optimization3d(f=f["f"], result=res, x_range=x_range, y_range=y_range,
                                     title=task["name"])
    visualizer.visual_trajectory(res, task["name"])
    visualizer.visual_minimizing(f["f"], res, task["name"])
    print(task["name"])
    print(f"Cтартовая  точка: {x0} -> конечная точка: {res[len(res) - 1]} -> {f['f'](res[len(res) - 1])}")
    if cnt["iter"] != 0:
        print(
            f'Всего   итераций: {cnt["iter"]}\nВызовов      "f": {cnt["f"]}\nВызовов "grad f": {cnt["df"]}\nТочек        "x": {cnt["x"]}\nПри eps={cnt["eps"]}\tmax_iters={cnt["max_iters"]}')


if __name__ == "__main__":
    q = functions.quadratic[0]
    visualizer.visual_heat_lines2d(q["f"], "0.6x^2 + 0.7y^2 + 0.12xy", q["x_range"], q["y_range"])
    x0 = functions.get_point(q["x_range"], q["y_range"])
    for algo in tasks:
        show_task(algo, q, x0)
        print()