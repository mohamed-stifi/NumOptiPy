import numpy as np
from numdifftools import Gradient , Hessian
from Steepest_descent import Fibonacci

# Solve a quadratic optimization problem Qx = b
def Conjugate_gradient_Q_Functions(Q, b, x0):
    """
    Solve a quadratic optimization problem Qx = b using the Conjugate Gradient method.

    Parameters:
        Q: Symmetric positive-definite matrix
        b: Vector
        x0: Initial guess

    Returns:
        xk: Solution vector
    """
    n = len(b)
    xk = x0
    gk =np.dot(Q, xk) - b
    dk = -gk
    oldNorm_gk = np.dot(gk,gk)
    for k in range(n):
        Q_dk = np.dot(Q,dk)
        al_k = oldNorm_gk/np.dot(Q_dk,dk)

        xk = xk + al_k*dk

        gk = np.dot(Q, xk) - b

        newNorm_gk = np.dot(gk,gk)
        beta_k = newNorm_gk/oldNorm_gk

        if k < n-1 :
            dk = -gk + beta_k*dk
            oldNorm_gk = newNorm_gk
    return xk

# Minimize a non-quadratic function
def Conjugate_gradient_Non_Quadratic_Functions(f, x0, tol= 1e-9,maxIter =10000):
    """
    Minimize a non-quadratic function using the Conjugate Gradient method.

    Parameters:
        f: Objective function to minimize
        x0: Initial guess
        tol: Tolerance for stopping criterion (default is 1e-9)
        maxIter: Maximum number of iterations (default is 10000)

    Returns:
        xk: Approximate minimizer
        path: Array containing the optimization path
    """
    Gf = Gradient(f); Hf = Hessian(f)
    n = len(x0)
    xk = x0
    gk = Gf(xk)
    iter_ = 0
    path = [xk]
    while iter_ <maxIter:
        dk = -gk
        for k in range(n):
            Q_dk = np.dot(Hf(xk),dk)
            dk_Q_dk = np.dot(Q_dk,dk)
            al_k = - np.dot(gk,dk)/dk_Q_dk
            xk = xk + al_k*dk

            gk = Gf(xk)

            beta_k = np.dot(gk,Q_dk)/dk_Q_dk
            if k < n-1 :
                dk = -gk + beta_k*dk
            
            path.append(xk)

        iter_ += 1 
        if np.linalg.norm(gk) < tol :
            return xk, np.array(path) 
    
    return xk, np.array(path)

# Minimize a non-quadratic function using Fletcher-Reeves formula
def Conjugate_gradient_Fletcher_Reeves(f, x0, tol= 1e-9,maxIter =10000):
    """
    Minimize a non-quadratic function using the Conjugate Gradient method with Fletcher-Reeves formula.

    Parameters:
        f: Objective function to minimize
        x0: Initial guess
        tol: Tolerance for stopping criterion (default is 1e-9)
        maxIter: Maximum number of iterations (default is 10000)

    Returns:
        xk: Approximate minimizer
        path: Array containing the optimization path
    """
    Gf = Gradient(f)
    n = len(x0)
    xk = x0
    gk = Gf(xk)
    dk = -gk
    oldNorm = np.dot(gk,gk)
    iter_ = 0
    path = [xk]
    def phi(al):
        return f(xk + al*dk)
    while iter_ < maxIter:
        dk = -gk
        for k in range(n):
            al_k = Fibonacci(phi, 0, 10, 40)
            xk = xk + al_k*dk

            gk = Gf(xk)
            newNorm = np.dot(gk,gk)

            if k < n-1 :
                beta_k = newNorm/oldNorm
                oldNorm = newNorm
                dk = -gk + beta_k*dk

            path.append(xk)
        iter_ += 1 
        if np.linalg.norm(gk) < tol :
            return xk , np.array(path) 
    
    return xk, np.array(path) 

# Minimize a non-quadratic function using Polak-Ribiere formula
def Conjugate_gradient_Polak_Ribiere(f, x0, tol= 1e-9,maxIter =10000):
    """
    Minimize a non-quadratic function using the Conjugate Gradient method with Polak-Ribiere formula.

    Parameters:
        f: Objective function to minimize
        x0: Initial guess
        tol: Tolerance for stopping criterion (default is 1e-9)
        maxIter: Maximum number of iterations (default is 10000)

    Returns:
        xk: Approximate minimizer
        path: Array containing the optimization path
    """
    Gf = Gradient(f)
    n = len(x0)
    xk = x0
    gk = Gf(xk)
    dk = -gk
    oldNorm = np.dot(gk,gk)
    iter_ = 0
    path = [xk]
    def phi(al):
        return f(xk + al*dk)
    while iter_ < maxIter:
        dk = -gk
        for k in range(n):
            al_k = Fibonacci(phi, 0, 10, 40)
            xk = xk + al_k*dk

            g0 = gk
            gk = Gf(xk)
            newNorm = np.dot(gk,gk)

            
            if k < n-1 :
                beta_k = (newNorm - np.dot(gk,g0))/oldNorm
                oldNorm =newNorm
                dk = -gk + beta_k*dk

            path.append(xk)

        iter_ += 1 
        if np.linalg.norm(gk) < tol :
            return xk, np.array(path)  
    
    return xk, np.array(path)

