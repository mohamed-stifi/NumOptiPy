import numpy as np
from numdifftools import Gradient, Hessian
from Steepest_descent import Fibonacci

def newton_method(f, x0, delta = 1e-8, max_iter = 10000):
    """
    Minimize a function using Newton's method.

    Parameters:
        f: Objective function to minimize
        x0: Initial guess
        delta: Tolerance for stopping criterion (default is 1e-8)
        max_iter: Maximum number of iterations (default is 10000)

    Returns:
        xk: Approximate minimizer
        path: Array containing the optimization path
    """
    k = 0
    alpha = .01
    xk = x0
    Gf = Gradient(f)
    Hf = Hessian(f)
    dk = np.dot(np.linalg.inv(Hf(xk)), Gf(xk))
    I = np.identity(len(x0), np.double)
    path = [xk]
    def phi(al):
        return f(xk - al*dk)

    while np.linalg.norm(dk) > delta and k < max_iter:
        alpha = Fibonacci(phi, 0, 10, 40)
        x0 = xk
        xk = xk - alpha*dk

        if np.all(np.linalg.eigvals(Hf(xk))>0):
            dk = np.dot(np.linalg.inv(Hf(xk)), Gf(xk))
        else :
            eps = 1e-6
            dk = np.dot(np.linalg.inv(I + Hf(x0)), Gf(x0))
        path.append(xk)
        k = k+1
    return xk, np.array(path)

def Armijo_rule(f, xk, dk, gk, al0, step = 2,eps = 0.2):
    """
    Armijo Rule for line search in optimization.

    Parameters:
    - f: Objective function.
    - xk: Current iterate.
    - dk: Search direction.
    - gk: Gradient at xk.
    - al0: Initial alpha value.
    - eps: Parameter in the range (0, 1).
    - step: Factor to increase alpha.

    Returns:
    - The chosen alpha value.
    """
    # Define the objective function phi(al)
    def phi(al):
        return f(xk + al*dk)
    
    # Define the function phiBar(al) for Armijo rule
    def  phiBar(al):
        return phi(0) + eps*al*np.dot(dk,gk)
    
    # Initial alpha value
    al = al0

    # Check the Armijo condition
    if (phi(al) <= phiBar(al)):
        # Increase alpha until the Armijo condition is satisfied
        while phi(al) <= phiBar(al):
            al = step*al
        # Return the alpha value that satisfies the condition
        return al/step
    else: 
        # Decrease alpha until the Armijo condition is satisfied        
        while phi(al) > phiBar(al):
            al = al/step 
        # Return the alpha value that satisfies the condition
        return al
    
def Quasi_newton(f, x0, eps = 1e-8, al0 = 1e3):
    """
    Quasi-Newton optimization using the DFP update.

    Parameters:
    - f: Objective function.
    - x0: Initial guess.
    - eps: Convergence tolerance.
    - al0: Initial alpha value for line search.

    Returns:
    - Optimal point.

    Notes:
    - The DFP update formula is used for updating the inverse Hessian matrix.
    """
    k = 0
    Gf = Gradient(f)
    xk = x0
    g0 = Gf(x0)
    gk = g0
    H = np.identity(len(x0),np.double) 
    while np.linalg.norm(g0) > eps :
        # Compute search direction
        dk = - np.dot(H,gk)

        # Perform line search using the Armijo rule
        al = armijoRule(f, xk, dk, gk, al0, step = 2,eps = 0.2) 
        x0 = xk
        xk = xk + al*dk

        g0 = gk
        gk = Gf(xk)

        # Determine D.F.P update parameters
        yk = gk - g0

        # Check if yk is not close to zero to avoid division by zero
        not_close_to_zero = ~np.isclose(yk, 0, atol= 1e-100)
        if np.any(not_close_to_zero):
            A = al* (np.outer(dk,dk)/np.dot(dk, yk))
            H_y = np.dot(H,yk)
            B = - np.outer(H_y,H_y)/np.dot(H_y, yk)
        else :

            # If yk is close to zero, return the current point
            return xk , k

        # Update the inverse Hessian matrix using the DFP formula
        H = H + A + B
        k = k + 1
    
    return xk , k