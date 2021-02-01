"""
Created on Fri Nov 13 10:57:55 2020

@author: Ng Boon Yong
"""

import matplotlib.pyplot as mpl

# Define the functions dy1, dy2, dy3 and dy4
def dy1(x, y1, y2):
    return y2

def dy2(x, y1, y2):
    return (2*y2 + y1 - x) / 7

def dy3(x, y3, y4):
    return y4

def dy4(x, y3, y4):
    return (2*y4 + y3) / 7

# Set the boundary conditions
x0, alph =  0, 5
xn, beta = 20, 8

# Choose the number of grid points
n = 10
h = (xn - x0) / n

# Create a list containing x0, x1, ..., xn
x = [x0 + i*h for i in range(n+1)]

# Define the RK4 function
def RK4(xi, y1i, y2i, F1, F2):
    '''
    Solves two coupled IVPs using the Runge-Kutta 4th order method

    Takes in 5 arguments:
        xi, y1i, y2i = Initial values x0, y1(x0) and y2(x0)
        F1, F2       = Two functions in the IVPs to be solved simultaneously

    Returns two lists containing the values of y1 and y2 at each iteration
    '''

    list1, list2 = [], []

    for i in range(n+1):
        list1.append(y1i)
        list2.append(y2i)

        K11 = h*F1(xi, y1i, y2i)
        K12 = h*F2(xi, y1i, y2i)

        K21 = h*F1(xi+0.5*h, y1i+0.5*K11, y2i+0.5*K12)
        K22 = h*F2(xi+0.5*h, y1i+0.5*K11, y2i+0.5*K12)

        K31 = h*F1(xi+0.5*h, y1i+0.5*K21, y2i+0.5*K22)
        K32 = h*F2(xi+0.5*h, y1i+0.5*K21, y2i+0.5*K22)

        K41 = h*F1(xi+h, y1i+K31, y2i+K32)
        K42 = h*F2(xi+h, y1i+K31, y2i+K32)

        y1i += (K11+2*K21+2*K31+K41)/6
        y2i += (K12+2*K22+2*K32+K42)/6
        xi  += h

    return list1, list2

# Run the RK4 function twice, once for y1 & y2, another once for y3 & y4
y1, y2 = RK4(x0, alph, 0, dy1, dy2)
y3, y4 = RK4(x0,    0, 1, dy3, dy4)

# Obtain the values of y1(b) and y3(b)
y1n = y1[-1]
y3n = y3[-1]

# Calculate the coefficient of y3(x)
c = (beta - y1n) / y3n

# Obtain the appoximate solution of the BVP, u(x) = y1(x) + c*y3(x)
u = [y1[i] + c*y3[i] for i in range(n+1)]

# Print the results
print(f'c = {c}\n')
print('{:15}{}\n{}'.format('    xi', 'yi', '-'*22))
for i in range(n+1):
    print('{:8.4f}{:12.6f}'.format(x[i], u[i]))

# Plot the graph
mpl.title ('Linear Shooting Method for BVP', fontsize=18)
mpl.xlabel('x'   , fontsize=16)
mpl.ylabel('u(x)', fontsize=16)
mpl.plot(x, u, 'ro')

mpl.savefig('CA2_Q1_Graph')
mpl.show()
