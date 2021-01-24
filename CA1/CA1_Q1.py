"""
Created on Thu Oct 29 01:10:43 2020

@author: Ng Boon Yong
"""

import numpy as np
import matplotlib.pyplot as mpl
import scipy.linalg as la

# Define the functions p, q and f
def p(x):
    return 0

def q(x):
    return np.pi**2

def f(x):
    return (2*np.pi**2) * np.cos(x*np.pi)

# Set the boundary conditions: u(x0) = alpha, u(xn) = beta
x0, alph = 0,  1
xn, beta = 1, -1

# Choose the number of grid points
n = 8
h = (xn - x0) / n

# Create a list containing x0, x1, ..., xn
x = [x0 + i*h for i in range(n+1)]

# Create empty lists and fill in the values of the coefficients
# These are generalized to the cases where p(x) and q(x) are functions of x
t1, t2, t3, fc = [], [], [], []
for i in range(1, n):
    t1.append(-p(x[i])*h /2 - 1)
    t3.append( p(x[i])*h /2 - 1)
    t2.append( q(x[i])*h**2 + 2)
    fc.append( f(x[i])*h**2    )

# Create an empty (n-1) x (n-1) array T, and a list F with the first entry
T = np.empty((n-1, n-1))
F = [fc[0] - (t1[0] * alph)]

# Fill in the main diagonal of T
for i in range(n-1):
    T[i][i] = t2[i]

# Fill in the subdiagonal and superdiagonal of T
for i in range(n-2):
    T[i+1][i] = t1[i+1]
    T[i][i+1] = t3[i]

# Fill in the rest of the entries of T with 0, and the middle entries of F
# The loops here are generalized for any integer value of n ≥ 3
for j in range(n-3):
    for i in range(n-j-3):
        T[i+j+2][i] = 0
        T[i][i+j+2] = 0
    F.append(fc[j+1])

# Append the last entry of F
F.append(fc[n-2] - (t3[n-2] * beta))

# Solve the matrix equation TU = F for U and print the result to 6 d.p.
u = list(la.solve(T, F))
U = [round(num, 6) for num in u]
print(f'h = 1/{n}\n')
print('U =', U)

# Include the values of alpha and beta to the list u and plot the graph
# This is to ensure that the lists x and u have the same dimensionality
u = [alph] + u + [beta]
mpl.plot(x, u, 'ro')

# Set the title and the axes
mpl.title ('Finite Difference Method for BVP', fontsize=18)
mpl.xlabel('x'   , fontsize=16)
mpl.ylabel('u(x)', fontsize=16)

mpl.savefig('CA1_Q1_Graph')
mpl.show()
