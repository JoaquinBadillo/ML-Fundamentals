# Import libraries
import numpy as np
from scipy import optimize

# Function
def f (x, y) :
    return -np.exp(x-y**2+x*y)

# Constraint
def g (x, y) :
    return np.cosh(y) + x - 2

# Derivatives
def dfdx (x, y) :
    return -np.exp(x-y**2+x*y)*(1+y)

def dfdy (x, y) :
    return np.exp(x-y**2+x*y)*(2*y-x)

def dgdx (x, y) :
    return 1

def dgdy (x, y) :
    return np.sinh(y)

# Lagrange Multiplier
def DL (xyλ) :
    [x, y, λ] = xyλ
    return np.array([
            dfdx(x, y) - λ * dgdx(x, y),
            dfdy(x, y) - λ * dgdy(x, y),
            - g(x, y)
        ])

x, y, λ = optimize.root(DL, [0, 0, 0]).x

print("x = %g" % x)
print("y = %g" % y)
print("λ = %g" % λ)
print("f(x, y) = %g" % f(x, y))