"""
Created on Tue Oct 27 20:00:22 2020

@author: Ng Boon Yong
"""

import matplotlib.pyplot as mpl
import scipy.linalg as la
import scipy.sparse as sp

# -y" + p(x)y' + q(x)y = f(x)
# Define the functions p, q and f
def p(x):
    return 2*x / (1 + x**2)

def q(x):
    return -2 / (1 + x**2)

def f(x):
    return -1

# Set the boundary conditions
x0, y0 = 0,  1.25
xn, yn = 4, -0.95

# Choose the number of grid points
n = 20
h = (xn - x0) / n
x = [x0 + i*h for i in range(n+1)]

# Create lists for the coefficients
c1, c2, c3 = [], [], []
for i in range(1, n):
    c1.append(-p(x[i])*h /2 - 1)
    c2.append( q(x[i])*h**2 + 2)
    c3.append( p(x[i])*h /2 - 1)

# Construct the column matrix B
B = [f(x[i])*h**2 for i in range(1, n)]
B[ 0] -= y0*c1[ 0]
B[-1] -= yn*c3[-1]

# Remove the first entry of c1 and the last entry of c3
c1 = c1[ 1:]
c3 = c3[:-1]

# Construct the tridiagonal matrix A
diag = [c1, c2, c3]
A = sp.diags(diag, [-1, 0, 1]).toarray()

# Solve the matrix equation Ay = B for y, add on y0 & yn
y = list(la.solve(A, B))
y = [y0] + y + [yn]

# Print the results
print('{:11}{}\n{}'.format(' xi', 'yi', '-'*16))
for i in range(n+1):
    print('{:.2f}{:12.6f}'.format(x[i], y[i]))

# Plot the graph
mpl.title ('Finite Difference Method for BVP', fontsize=18)
mpl.xlabel('x'   , fontsize=16)
mpl.ylabel('y(x)', fontsize=16)
mpl.plot(x, y, 'r.')
mpl.show()
