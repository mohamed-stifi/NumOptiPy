"""
NumOptiPy

This package provides implementations of various numerical methods and algorithms for solving mathematical problems,
including elimination methods, root-finding methods, matrix factorization, optimization (gradient descent),
conjugate gradient methods, and Newton's method.

Modules:
- Elimination_methods: Module containing functions for optimization and search algorithms.
- Root_finding_methods: Module with implementations of root-finding methods such as Newton-Raphson, Quasi-Newton, and Secant.
- Matrix_factorisation_inverse: Module for matrix factorization and solving linear systems using LU decomposition, Cholesky decomposition, and Gauss elimination.
- Steepest_descent: Module for optimization using gradient descent.
- Conjugate_gradient: Module with implementations of Conjugate Gradient methods.
- Newton: Module containing implementations of Newton's method and related algorithms.

Note: This package is intended for educational purposes and may not be suitable for all use cases.
"""

# Importing functions from modules
from Elimination_methods import search_with_fixe_step, search_with_accelerat_step,\
                                exhaustive_Search,dichotomous_Search, interval_Halving,\
                                Fibonacci, golden_Section
from Root_finding_methods import Newton_Rapson, Quasi_Newton, Secant
from Matrix_factorisation_inverse import for_Elimination, back_Substitution, Gauss_solve,\
                                        LU, PLU, Cholesky_decomposition, Solve_lower_system,\
                                        Solve_upper_system, Solve_by_LU, Solve_by_PLU,\
                                        LU_inv, Cholesky_inv, Gauss_inv
from Steepest_descent import Gradient_descent, fixed_step_gradient_descent, creat_animation
from Conjugate_gradient import Conjugate_gradient_Fletcher_Reeves, Conjugate_gradient_Q_Functions,\
                            Conjugate_gradient_Non_Quadratic_Functions, Conjugate_gradient_Polak_Ribiere
from Newton import newton_method, Quasi_newton, Armijo_rule
