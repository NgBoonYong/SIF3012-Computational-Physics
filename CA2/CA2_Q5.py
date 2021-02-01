"""
Created on Fri Nov 13 12:36:41 2020

@author: Ng Boon Yong
"""

import numpy as np
import matplotlib.pyplot as mpl
import scipy.linalg as la
import scipy.sparse as sp

# u" = f(x, u, u')
# Define f, ∂f/∂u, ∂f/∂u' and the exact solution
def f(x, u, du):
    return -du**2

def fu(x, u, du):
    return 0

def fdu(x, u, du):
    return -2*du

def uex(x):
    return np.log((np.exp(1)-1)*x + 1)

# Set the boundary conditions
x0, alph = 0, 0
xn, beta = 1, 1

# Set the grid points
n  = 20
h  = (xn - x0) / n

# Create a list containing x0, x1, ..., xn, and calculate u(xi)
x  = [x0 + i*h for i in range(n+1)]
ux = [uex(val) for val in x]

# Initial approximation
m  = (beta - alph) / (xn - x0)
w  = [alph + m*(x[i] - x0) for i in range(n+1)]
wl = [w]

# Set the tolerance and the max. no. of iterations
k, lim, tol = 0, 20, 1e-8
while k < lim:
    # Create lists for the coefficients, and the matrix F
    j1, j2, j3, F = [], [], [], []
    for i in range(1, n):
        d = (wl[k][i+1] - wl[k][i-1]) / (2*h)
        j1.append(-fdu(x[i], wl[k][i], d)*h /2 - 1)
        j2.append( fu (x[i], wl[k][i], d)*h**2 + 2)
        j3.append( fdu(x[i], wl[k][i], d)*h /2 - 1)
        F .append( wl[k][i-1]-2*wl[k][i]+wl[k][i+1]-f(x[i], wl[k][i], d)*h**2)

    # Construct the tridiagonal matrix J
    j1 = j1[ 1:]
    j3 = j3[:-1]
    dg = [j1, j2, j3]
    J  = sp.diags(dg, [-1, 0, 1]).toarray()

    # Solve the matrix equation Jv = F for v
    v  = list(la.solve(J, F))

    # Next approximation
    wp = [alph] + [wl[k][i+1] + v[i] for i in range(len(v))] + [beta]
    wl.append(wp)

    # Check the tolerance condition
    if all(abs(wl[k+1][i] - wl[k][i]) < tol for i in range(len(wl[k]))):
        print(f'Number of iterations: {k+1}\n')
        break

    elif k+1 == lim:
        print(f'Maximum number of iterations ({lim}) reached\n')
        break

    k += 1

# Print the results
u = wl[-1]
print('{:13}{:11}{}\n{}'.format('  xi', 'ui', 'Error', '-'*30))
for i in range(n+1):
    print('{:.4f}{:12.6f}{:12.6f}'.format(x[i], u[i], u[i]-ux[i]))

# Plot the graph
xl = np.linspace(x0, xn, 50)
mpl.title ('Nonlinear Finite Difference Method', fontsize=18)
mpl.xlabel('x'   , fontsize=16)
mpl.ylabel('u(x)', fontsize=16)
mpl.plot(x , u      , 'k.')
mpl.plot(xl, uex(xl), 'r-')

mpl.savefig('CA2_Q5_Graph')
mpl.show()
