import numpy as np

# forward elimination
def for_Elimination(A,b):
    """
    Parameters:
        A: Coefficient matrix
        b: Right-hand side vector

    Returns:
        U: Upper triangular matrix after Gaussian elimination
        y: Transformed right-hand side vector
    """

    n = A.shape[0]
    arr = np.zeros((n, n+1))
    arr[:,:n] = A.copy()
    arr[:,n] = b.copy() 
    for i in range(n-1):
        p_ind = i + np.argmax(np.abs(arr[i:,i]))
        arr[i,i:], arr[p_ind,i:] = arr[p_ind,i:].copy(),arr[i,i:].copy()
        pivot = arr[i,i]
        if pivot == 0 : 
            print("The system does not have a unique solution.")
        for k in range(i+1, n):
            m = arr[k,i]/pivot
            arr[k,i:] -= m*arr[i, i:]
    U, y = arr[:, :n], arr[:, n]
    return U, y
    
# back substitution
def back_Substitution(A,b):
    """
    Parameters:
        A: Upper triangular matrix
        b: Right-hand side vector

    Returns:
        X: Solution vector
    """
    n = A.shape[0]
    X = np.empty(n)
    X[-1] = b[-1]/A[-1,-1]
    for i in range(-2, -(n+1), -1):
        X[i] = (b[i] - np.dot(A[i,i+1:], X[i+1:]))/A[i,i]
    
    return X

# Solve a system of linear equations
def Gauss_solve(A, b):
    """
    Solve a system of linear equations Ax = b using Gaussian elimination with partial pivoting.
    
    Parameters:
        A: Coefficient matrix
        b: Right-hand side vector
    
    Returns:
        x: Solution vector
    """
    a, bb = for_Elimination(A, b)  
    x = back_Substitution(a, bb)  
    return x

# Implementation for the LU decomposition
def LU(A) :
    """
    LU decomposition of a square matrix A into lower triangular matrix L and upper triangular matrix U.
    
    Parameters:
        A: Square matrix to be decomposed
        

    Returns:
        L: Lower triangular matrix
        U: Upper triangular matrix
    """

    n = len(A)
    U = A.copy()
    L = np.eye(n)
    for i in range(n-1):

        pivot =  U[i,i]
        if pivot == 0 : 
            print("Pivot is 0, LU decomposition may not be unique.")
            return L,U

        for k in range(i+1, n):
            '''if ameliorer:
                # Partial pivoting
                pivot_index = np.argmax(np.abs(U[i:, i])) + i
                U[[i, pivot_index]] = U[[pivot_index, i]]
                L[[i, pivot_index], :i] = L[[pivot_index, i], :i]

            L[k, i] = U[k, i] / pivot
            U[k, i:] -= L[k, i] * U[i, i:]'''
            L[k,i] = U[k,i]/pivot
            U[k] = U[k] - L[k,i]*U[i]
    return L,U

# PLU decomposition
def PLU(A):
    """
    PLU decomposition of a square matrix A into permutation matrix P, lower triangular matrix L, and upper triangular matrix U.

    Parameters:
        A: Square matrix to be decomposed

    Returns:
        P: Permutation matrix
        L: Lower triangular matrix
        U: Upper triangular matrix
    """

    n = A.shape[0]
    U = A.copy()
    L = np.eye(n, dtype= np.double)
    P = np.eye(n, dtype= np.double)

    for i in range(n):
        for k in range(i,n):
            if ~np.isclose(U[i,i], 0) :
                break
            U[[k,k+1]] = U[[k+1, k]]
            P[[k,k+1]] = P[[k+1, k]]

        for k in range(i+1, n):
            L[k,i] = U[k,i]/U[i,i]
            U[k] = U[k] - L[k,i]*U[i]
    return P,L,U

# Cholesky decomposition
def Cholesky_decomposition(A):
    """
    Parameters:
        A: Positive definite matrix for Cholesky decomposition

    Returns:
        L: Lower triangular matrix such that A = L * L^T
    """
    L = np.zeros(A.shape, dtype= np.double)
    n = A.shape[0]
    for k in range(n):
        L[k,k] = np.sqrt(A[k,k] - np.dot(L[k,:k], L[k,:k]))

        for i in range(k+1, n):
            L[i,k] = ( A[i,k] - np.dot(L[i,:k], L[:k,k]) )/L[k,k]
        
    return L

# Solve a lower triangular system
def Solve_lower_system(L,b):
    """
    Solve a lower triangular system of linear equations Lz = b.

    Parameters:
        L: Lower triangular matrix
        b: Right-hand side vector

    Returns:
        Z: Solution vector
    """

    n = L.shape[0]
    Z = np.zeros(n, dtype= np.double)
    Z[0] = b[0]/L[0,0]
    for i in range(1,n):
        Z[i] = (b[i] - np.dot(L[i,:i], Z[:i]))/L[i,i]
    return Z

# Solve an upper triangular system
def Solve_upper_system(U,z):
    """
    Solve an upper triangular system of linear equations Ux = z.

    Parameters:
        U: Upper triangular matrix
        z: Right-hand side vector

    Returns:
        X: Solution vector
    """
    n = U.shape[0]
    X = np.zeros(n, dtype= np.double)
    X[-1] = z[-1]/U[-1,-1]
    for i in range(n-2, -1, -1):
        X[i] = (z[i] - np.dot(U[i,i+1:], X[i+1:]) )/U[i,i]
    return X

# Solve a system of linear equations
def Solve_by_LU(A, b):
    """
    Solve a system of linear equations Ax = b using LU decomposition.

    Parameters:
        A: Coefficient matrix
        b: Right-hand side vector

    Returns:
        x: Solution vector
    """
    L, U = LU(A)
    z = Solve_lower_system(L, b)  
    x = Solve_upper_system(U, z)  
    return x

def Solve_by_PLU(A, b):
    """
    Solve a system of linear equations Ax = b using PLU decomposition.

    Parameters:
        A: Coefficient matrix
        b: Right-hand side vector

    Returns:
        x: Solution vector
    """
    P, L, U = PLU(A)
    z = Solve_lower_system(L, np.dot(P,b))  
    x = Solve_upper_system(U, z)  
    return x

# Compute the inverse of a matrix using LU decomposition
def LU_inv(L,U, P = None):
    """
    Compute the inverse of a matrix using LU decomposition.

    Parameters:
        L: Lower triangular matrix from LU decomposition
        U: Upper triangular matrix from LU decomposition
        P: Permutation matrix (optional)

    Returns:
        Inv: Inverse matrix
    """

    n = L.shape[0]
    Inv = np.empty((n,n), dtype=np.double)
    x = np.empty(n, dtype=np.double)
    b = np.zeros(n, dtype=np.double)
    for i in range(n):
        b[i] = 1
        if P is not None:
            x = Solve_lower_system(L,np.dot(P,b))
        else :
            x = Solve_lower_system(L, b)
        x = Solve_upper_system(U, x)
        Inv[:,i] = x
        b[i] = 0
    return Inv

# Compute the inverse of a matrix using Cholesky decomposition.
def Cholesky_inv(A):
    """
    Compute the inverse of a matrix using Cholesky decomposition.

    Parameters:
        A: Positive definite matrix for Cholesky decomposition

    Returns:
        Inv: Inverse matrix
    """
    L = Cholesky_decomposition(A)
    n = L.shape[0]
    Inv = np.empty((n,n), dtype=np.double)
    x = np.empty(n, dtype=np.double)
    b = np.zeros(n, dtype=np.double)
    for i in range(n):
        b[i] = 1
        x = Solve_lower_system(L, b)
        x = Solve_upper_system(L.T, x)
        Inv[:,i] = x
        b[i] = 0
    return Inv

# Compute the inverse of a matrix using Gaussian elimination
def Gauss_inv(A):
    """
    Compute the inverse of a matrix using Gaussian elimination with partial pivoting.

    Parameters:
        A: Coefficient matrix

    Returns:
        inv_A: Inverse matrix
    """
    n = A.shape[0]
    inv_A = np.empty((n,n))
    b = np.array([0]*n)
    for i in range(n):
        b[i] = 1
        inv_A[:,i] = Gauss_solve(A,b)
        b[i] = 0
        
    return inv_A
