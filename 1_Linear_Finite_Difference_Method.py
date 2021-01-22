"""
Created on Tue Nov 10 20:00:22 2020

@author: Ng Boon Yong
"""

import matplotlib.pyplot as mpl
import scipy.linalg as la
import scipy.sparse as sp

# Define the functions p, q and f in -u" + p(x)u' + q(x)u = f(x)
def p(x):
    return 2*x / (1 + x**2)

def q(x):
    return -2 / (1 + x**2)

def r(x):
    return -1

# Set the boundary conditions
x0, u0 = 0,  1.25
xn, un = 4, -0.95

# Choose the number of grid points
n = 20
h = (xn - x0) / n
x = [x0 + i*h for i in range(n+1)]

# Create empty lists and fill in the coefficients
c1, c2, c3 = [], [], []
for i in range(1, n):
    c1.append(-p(x[i])*h /2 - 1)
    c2.append( q(x[i])*h**2 + 2)
    c3.append( p(x[i])*h /2 - 1)

# Construct the column matrix B
B = [r(x[i])*h**2 for i in range(1, n)]
B[ 0] -= u0*c1[ 0]
B[-1] -= un*c3[-1]

# Remove the first entry of c1 and the last entry of c3
c1 = c1[ 1:]
c3 = c3[:-1]

# Construct the tridiagonal matrix A
diag = [c1, c2, c3]
A = sp.diags(diag, [-1, 0, 1]).toarray()

# Solve the matrix equation Au = B for u, add on u0 & un
u = list(la.solve(A, B))
u = [u0] + u + [un]

# Print the results
print('{:11}{}\n{}'.format(' xi', 'ui', '-'*16))
for i in range(n+1):
    print('{:.2f}{:12.6f}'.format(x[i], u[i]))

# Plot the graph
mpl.title ('Finite Difference Method for BVP', fontsize=18)
mpl.xlabel('x'   , fontsize=16)
mpl.ylabel('u(x)', fontsize=16)
mpl.plot(x, u, 'r.')
mpl.show()
