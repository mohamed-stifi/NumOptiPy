import numpy as np

# Newton-Rapson method
def Newton_Rapson( df , d2f, x0,num_iter_max= 1000, eps = 1e-9) :
    """
    Parameters:
        df: First derivative of the function
        d2f: Second derivative of the function
        x0: Initial guess for the root
        num_iter_max: Maximum number of iterations (default is 1000)
        eps: Tolerance for convergence (default is 1e-9)

    Returns:
        x0: Approximation of the root
        iter_: Number of iterations performed
    """
    f1  = df(x0)
    iter_ = 0
    while np.abs(f1) > eps and num_iter_max > iter_ :
        f2 = d2f(x0)
        if f2 != 0 :
            x0 -= f1/f2
            f1  = df(x0)
            iter_ += 1
        else :
            x0 -= f1/eps
            f1  = df(x0)
            iter_ += 1

    return x0 , iter_

# Quasi Newton Method
def Quasi_Newton(f, x0, delta= 1e-3,num_iter_max = 1000, eps = 1e-8):
    """
    Parameters:
        f: Objective function to be minimized
        x0: Initial guess for the minimum
        delta: Step size for finite difference approximation of the gradient(default is 1e-3)
        num_iter_max: Maximum number of iterations (default is 1000)
        eps: Tolerance for convergence (default is 1e-8)

    Returns:
        x0: Approximation of the minimum
        iter_: Number of iterations performed
    """
    f1 = f(x0 + delta)
    f2 = f(x0 - delta)
    delta_2 = 2*delta
    df = (f1 - f2)/delta_2
    iter_ = 0
    while np.abs(df) > eps and num_iter_max > iter_ :
        x0 -= (delta*(f1 - f2))/(2*(f1 - 2*f(x0) + f2))
        f1 = f(x0 + delta)
        f2 = f(x0 - delta)
        df = (f1 - f2)/delta_2
        iter_ += 1
    return x0 , iter_

# Secant Method
def Secant(df, a, b, num_iter_max = 1000, eps = 1e-8):
    """
    Parameters:
        df: Derivative of the function
        a: First initial guess
        b: Second initial guess
        num_iter_max: Maximum number of iterations (default is 1000)
        eps: Tolerance for convergence (default is 1e-8)

    Returns:
        x: Approximation of the root
        iter_: Number of iterations performed
    """
    dfx = df(a)
    x = a 
    iter_ = 0
    while np.abs(dfx) > eps and num_iter_max > iter_ :
        df1 = df(a)
        df2 = df(b)
        x = a - (df1*(b- a))/(df2 - df1)
        dfx = df(x)
        if dfx >= 0 :
            b = x
            df2 = dfx
        else :
            a = x
            df1 = dfx
        iter_ += 1
    return x , iter_
