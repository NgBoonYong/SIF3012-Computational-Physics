"""
Created on Fri Nov 13 10:57:55 2020

@author: Ng Boon Yong
"""

import numpy as np
import matplotlib.pyplot as mpl

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

# Calculate the initial guess, s0
s  = (yn - y0) / (xn - x0)

# Set the tolerance and the max. no. of iterations
k, lim, tol = 0, 20, 1e-08
while k < lim:
    # Initial values
    u1, u2 = [y0], [s]
    v1, v2 = 0, 1

    for i in range(n):
        # Runge-Kutta 4th order method
        k11 = h*u2[i]
        k12 = h*f(x[i], u1[i], u2[i])
        k21 = h*(u2[i]+k12/2)
        k22 = h*f(x[i]+h/2, u1[i]+k11/2, u2[i]+k12/2)
        k31 = h*(u2[i]+k22/2)
        k32 = h*f(x[i]+h/2, u1[i]+k21/2, u2[i]+k22/2)
        k41 = h*(u2[i]+k32)
        k42 = h*f(x[i]+h, u1[i]+k31, u2[i]+k32)

        u1.append(u1[i] + (k11+2*k21+2*k31+k41)/6)
        u2.append(u2[i] + (k12+2*k22+2*k32+k42)/6)

        j11  = h*v2
        j12  = h*v1*fy (x[i], u1[i], u2[i])
        j12 += h*v2*fdy(x[i], u1[i], u2[i])
        j21  = h*(v2+j12/2)
        j22  = h*(v1+j11/2)*fy (x[i]+h/2, u1[i], u2[i])
        j22 += h*(v2+j12/2)*fdy(x[i]+h/2, u1[i], u2[i])
        j31  = h*(v2+j22/2)
        j32  = h*(v1+j21/2)*fy (x[i]+h/2, u1[i], u2[i])
        j32 += h*(v2+j22/2)*fdy(x[i]+h/2, u1[i], u2[i])
        j41  = h*(v2+j32)
        j42  = h*(v1+j31)*fy (x[i]+h, u1[i], u2[i])
        j42 += h*(v2+j32)*fdy(x[i]+h, u1[i], u2[i])

        v1 += (j11+2*j21+2*j31+j41)/6
        v2 += (j12+2*j22+2*j32+j42)/6

    # Check if the approximation is sufficiently close to beta
    if abs(u1[n] - yn) <= tol:
        print(f'Number of iterations: {k+1}\n')
        break

    elif k+1 == lim:
        print(f'Maximum number of iterations ({lim}) reached\n')
        break

    # Use the Newton method for the next s
    s -= (u1[n] - yn) / v1
    k += 1

# Print the results
print('{:13}{:11}{}\n{}'.format('  xi', 'yi', 'Error', '-'*30))
for i in range(n+1):
    print('{:.4f}{:12.6f}{:12.6f}'.format(x[i], u1[i], u1[i]-yx[i]))

# Plot the graph
xl = np.linspace(x0, xn, 50)
mpl.title ('Nonlinear Shooting Method', fontsize=18)
mpl.xlabel('x'   , fontsize=16)
mpl.ylabel('y(x)', fontsize=16)
mpl.plot(x , u1     , 'k.')
mpl.plot(xl, yex(xl), 'c-')
mpl.show()
