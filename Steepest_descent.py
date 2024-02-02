import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 


def Fib(n):
    if n == 0 :
        return [1]
    f = [1,1]
    for i in range(2, n+1):
        f.append( f[-2] + f[-1])
    return f

def Fibonacci(f, x_l, x_u, n):
    fib = Fib(n)
    x1 = x_l + (fib[n-2]/ fib[n])*(x_u - x_l)
    x2 = x_l + (fib[n-1]/ fib[n])*(x_u - x_l)
    while n>1 :
        n = n - 1
        f1 = f(x1)
        f2 = f(x2)
        if f1 < f2 :
            x_u = x2
            x2 = x1
            x1 = x_l + (fib[n-2]/ fib[n])*(x_u - x_l)
            f2 = f1
            f1 = f(x1)
        elif f1 > f2 :
            x_l = x1
            x1 = x2
            f1 = f2
            x2 = x_l + (fib[n-1]/ fib[n])*(x_u - x_l)
            f2 = f(x2)
        else :
            x_l = x_l + 0.03*(x_u - x_l)
            x1 = x_l + (fib[n-2]/ fib[n])*(x_u - x_l)
            x2 = x_l + (fib[n-1]/ fib[n])*(x_u - x_l)
    return min([x_l , x_u], key = f)

def fixed_step_gradient_descent(df,x0, lr =0.01 ,eps =1e-9 ):
    """
    Gradient Descent a fixed step optimization algorithm.

    Parameters:
        f: Objective function to minimize
        df: Gradient of the objective function
        x0: Initial guess
        lr: Learning rate (default is 0.01)
        eps: Tolerance for stopping criterion (default is 1e-9)

    Returns:
        xk: Approximate minimizer
        path: Array containing the optimization path
    """
    xk = x0 
    xk1 = x0+ 2*eps
    dfxk = df(xk)
    path = [xk] 
    while np.linalg.norm(dfxk) > eps and np.linalg.norm(xk-xk1) > eps  :
        xk1 = xk
        xk = xk - lr*dfxk
        dfxk = df(xk)
        path.append(xk)
    return xk, np.array(path)

def Gradient_descent(f,df,x0, eps =1e-9 ):
    """
    Gradient Descent optimization algorithm with Fibonacci line search.

    Parameters:
        f: Objective function to minimize
        df: Gradient of the objective function
        x0: Initial guess
        eps: Tolerance for stopping criterion (default is 1e-9)

    Returns:
        xk: Approximate minimizer
        path: Array containing the optimization path
    """
    xk = x0 
    xk1 = x0+ 2*eps
    dfxk = df(xk)
    path = [xk] 
    def phi(al):
        return f(xk - al*dfxk)
    
    while np.linalg.norm(dfxk) > eps and np.linalg.norm(xk-xk1) > eps  :
        al = Fibonacci(phi, 0, 10, 40)
        xk1 = xk
        xk = xk - al*dfxk
        dfxk = df(xk)
        path.append(xk)
    return xk, np.array(path)

def creat_animation(f, paths, minimum, x_lim, y_lim, colors,
                    labels, n_seconds=7,figsize = (14,16)):
    try :
        path_length = max(len(path) for path in paths)
        n_points = 300
        x = np.linspace(*x_lim, n_points)
        y = np.linspace(*y_lim, n_points)
        X, Y = np.meshgrid(x, y)
        Z = f([X,Y])
        fig, ax = plt.subplots(figsize=figsize)
        ax.contour(X, Y, Z, 90, cmap="jet")
        scatters = [ax.scatter(None,
                            None,
                            label=label,
                            c=c) for c, label in zip(colors, labels)]
        ax.legend(prop={"size": 25})
        
        ax.plot(*minimum, "rD")
        
        def animate(i):
            for path, scatter in zip(paths, scatters):
                scatter.set_offsets(path[:i, :])

            ax.set_title(str(i))
        ms_per_frame = 1000 * n_seconds / path_length
        anim = FuncAnimation(fig, animate, frames=path_length, interval=ms_per_frame)
        plt.show()
    except Exception as e :
        print(e)
    return anim
