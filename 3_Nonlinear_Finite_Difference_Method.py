"""
Created on Thu Nov 12 17:44:09 2020

@author: Ng Boon Yong
"""

import numpy as np
import matplotlib.pyplot as mpl
import scipy.linalg as la
import scipy.sparse as sp

# y" = f(x, y, y')
# Define f, ∂f/∂y, ∂f/∂y' and the exact solution
def f(x, y, dy):
    return (32 + 2*x**3 - y*dy) / 8

def fy(x, y, dy):
    return -dy/8

def fdy(x, y, dy):
    return -y/8

def yex(x):
    return x**2 + 16/x

# Set the boundary conditions
x0, y0 = 1, 17
xn, yn = 3, 43/3

# Set the grid points
n  = 20
h  = (xn - x0) / n
x  = [x0 + i*h for i in range(n+1)]
yx = [yex(val) for val in x]

# Initial approximation
m  = (yn - y0) / (xn - x0)
w  = [y0 + m*(x[i] - x0) for i in range(n+1)]
wl = [w]

# Set the tolerance and the max. no. of iterations
k, lim, tol = 0, 20, 1e-8
while k < lim:
    # Create lists for the coefficients, and the matrix F
    j1, j2, j3, F = [], [], [], []
    for i in range(1, n):
        d = (wl[k][i+1] - wl[k][i-1]) / (2*h)
        j1.append(-fdy(x[i], wl[k][i], d)*h /2 - 1)
        j2.append( fy (x[i], wl[k][i], d)*h**2 + 2)
        j3.append( fdy(x[i], wl[k][i], d)*h /2 - 1)
        F .append( wl[k][i-1]-2*wl[k][i]+wl[k][i+1]-f(x[i], wl[k][i], d)*h**2)

    # Construct the tridiagonal matrix J
    j1 = j1[ 1:]
    j3 = j3[:-1]
    dg = [j1, j2, j3]
    J  = sp.diags(dg, [-1, 0, 1]).toarray()

    # Solve the matrix equation Jv = F for v
    v  = list(la.solve(J, F))

    # Next approximation
    wp = [y0] + [wl[k][i+1] + v[i] for i in range(len(v))] + [yn]
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
y = wl[-1]
print('{:13}{:11}{}\n{}'.format('  xi', 'yi', 'Error', '-'*30))
for i in range(n+1):
    print('{:.4f}{:12.6f}{:12.6f}'.format(x[i], y[i], y[i]-yx[i]))

# Plot the graph
xl = np.linspace(x0, xn, 50)
mpl.title ('Nonlinear Finite Difference Method', fontsize=18)
mpl.xlabel('x'   , fontsize=16)
mpl.ylabel('y(x)', fontsize=16)
mpl.plot(x , y      , 'k.')
mpl.plot(xl, yex(xl), 'r-')
mpl.show()
