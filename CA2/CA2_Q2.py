"""
Created on Fri Nov 13 12:28:35 2020

@author: Ng Boon Yong
"""

import matplotlib.pyplot as mpl
import scipy.linalg as la
import scipy.sparse as sp

# -u" + p(x)u' + q(x)u = f(x)
# Define the functions p, q and f
def p(x):
    return 2/7

def q(x):
    return 1/7

def f(x):
    return x/7

# Set the boundary conditions
x0, alph =  0, 5
xn, beta = 20, 8

# Choose the number of grid points
n = 10
h = (xn - x0) / n

# Create a list containing x0, x1, ..., xn
x = [x0 + i*h for i in range(n+1)]

# Create lists for the coefficients
c1, c2, c3 = [], [], []
for i in range(1, n):
    c1.append(-p(x[i])*h /2 - 1)
    c2.append( q(x[i])*h**2 + 2)
    c3.append( p(x[i])*h /2 - 1)

# Construct the column matrix B
B = [f(x[i])*h**2 for i in range(1, n)]
B[ 0] -= alph*c1[ 0]
B[-1] -= beta*c3[-1]

# Remove the first entry of c1 and the last entry of c3
c1 = c1[ 1:]
c3 = c3[:-1]

# Construct the tridiagonal matrix A
diag = [c1, c2, c3]
A = sp.diags(diag, [-1, 0, 1]).toarray()

# Solve the matrix equation Ay = B for y, add on alpha & beta
u = list(la.solve(A, B))
u = [alph] + u + [beta]

# Print the results
print('{:15}{}\n{}'.format('    xi', 'ui', '-'*22))
for i in range(n+1):
    print('{:8.4f}{:12.6f}'.format(x[i], u[i]))

# Plot the graph
mpl.title ('Finite Difference Method for BVP', fontsize=18)
mpl.xlabel('x'   , fontsize=16)
mpl.ylabel('u(x)', fontsize=16)
mpl.plot(x, u, 'ro')

mpl.savefig('CA2_Q2_Graph')
mpl.show()
