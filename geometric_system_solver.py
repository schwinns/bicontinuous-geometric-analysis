# Import packages
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Define the triply periodic minimal surface functions and their gradients
def SchwarzD(X, period):

    N = 2*np.pi/period
    
    a = np.sin(N*X[0]) * np.sin(N*X[1]) * np.sin(N*X[2])
    b = np.sin(N*X[0]) * np.cos(N*X[1]) * np.cos(N*X[2])
    c = np.cos(N*X[0]) * np.sin(N*X[1]) * np.cos(N*X[2])
    d = np.cos(N*X[0]) * np.cos(N*X[1]) * np.sin(N*X[2])
    
    return a + b + c + d

def Gyroid(X,period):
    
    N = 2*np.pi/period
    
    a = np.sin(N*X[0]) * np.cos(N*X[1])
    b = np.sin(N*X[1]) * np.cos(N*X[2])
    c = np.sin(N*X[2]) * np.cos(N*X[0])
    
    return a + b + c

def SchwarzD_grad(v,period):
    
    x = v[0]; y = v[1]; z = v[2]
    N = 2*np.pi / period
    
    a = N*np.cos(N*x)*np.sin(N*y)*np.sin(N*z) + N*np.cos(N*x)*np.cos(N*y)*np.cos(N*z) - N*np.sin(N*x)*np.sin(N*y)*np.cos(N*z) - N*np.sin(N*x)*np.cos(N*y)*np.sin(N*z)
    b = N*np.sin(N*x)*np.cos(N*y)*np.sin(N*z) - N*np.sin(N*x)*np.sin(N*y)*np.cos(N*z) + N*np.cos(N*x)*np.cos(N*y)*np.cos(N*z) - N*np.cos(N*x)*np.sin(N*y)*np.sin(N*z)
    c = N*np.sin(N*x)*np.sin(N*y)*np.cos(N*z) - N*np.sin(N*x)*np.cos(N*y)*np.sin(N*z) - N*np.cos(N*x)*np.sin(N*y)*np.sin(N*z) + N*np.cos(N*x)*np.cos(N*y)*np.cos(N*z)
    
    return np.array([a,b,c]) / np.linalg.norm(np.array([a,b,c]))

def Gyroid_grad(v,period):
    
    x = v[0]; y = v[1]; z = v[2]
    N = 2*np.pi / period
    
    a =  N*np.cos(N*x)*np.cos(N*y) - N*np.sin(N*x)*np.sin(N*z)
    b = -N*np.sin(N*y)*np.sin(N*x) + N*np.cos(N*y)*np.cos(N*z)
    c = -N*np.sin(N*y)*np.sin(N*z) + N*np.cos(N*z)*np.cos(N*x)
    
    return np.array([a,b,c]) / np.linalg.norm(np.array([a,b,c]))

def Primitive(X, period):
    
    N = 2*np.pi/period
    
    a = np.cos(N*X[0]) + np.cos(N*X[1]) + np.cos(N*X[2])
    
    return a

def Primitive_grad(v, period):
    
    x = v[0]; y = v[1]; z = v[2]
    N = 2*np.pi / period
    
    a = -N*np.sin(N*x) 
    b = -N*np.sin(N*y) 
    c = -N*np.sin(N*z)
    
    return np.array([a,b,c]) / np.linalg.norm(np.array([a,b,c]))

def P_gyroid(t,X,period=9.4): # return point extended along the normal from gyroid surface
    n = Gyroid_grad(X,period)
    
    return X + t*n

def P_schwarz(t,X,period=9.4): # return point extended along the normal from schwarz surface
    n = SchwarzD_grad(X,period)
    
    return X + t*n

def P_primitive(t,X,period=9.4): # return point extended along the normal from primitive surface
    n = Primitive_grad(X,period)
    
    return X + t*n


def Gyroid_eq(solution, params):

    '''
    Function to give to fsolve to solve system of equations

    Input params is dictionary with variables to solve for (alpha, x, y, z) and input constants (period, x0, y0, z0)
    '''
    

    # input constants
    period = params['period'] # period for gyroid
    x0, y0, z0 = params['point'] # point on the surface to find the distance to surface again

    # variables to solve for
    alpha, x, y, z = solution # point on the surface when projecting normal from [x0,y0,z0] and the distance to project normal
    
    N = 2*np.pi / period
    F = Gyroid_grad([x0,y0,z0], period)

    eq1 = x - x0 - alpha*F[0]
    eq2 = y - y0 - alpha*F[1]
    eq3 = z - z0 - alpha*F[2]
    eq4 = Gyroid([x,y,z], period)

    return (eq1, eq2, eq3, eq4)


# Main function for generating the distance distributions
def surface2surface(structure,struct='gyroid',guess_dist=4.6,box=9.4,period=9.4,sample=10):

    # some hard coded constants
    short_dist_tol = 0.01
    small_tol = 0.01

    # intialize all variables
    distribution = np.zeros((sample))
    new_point = np.zeros((3))

    # define necessary functions for the chosen structure
    if struct == 'gyroid':
        P = P_gyroid
        solve_eq = Gyroid_eq
    elif struct == 'schwarzD':
        P = P_schwarz
        solve_eq = SchwarzD_eq
    elif struct == 'primitive':
        P = P_primitive
        solve_eq = Primitive_eq

    # generate distribution of surface to surface distances
    for i in range(len(distribution)):

        p = random.randint(0, structure.shape[0] - 1) # choose a random point on the discretized surface
        point = structure[p,:]

        # find the point actually on the surface
        params = {
            'period' : period,
            'point'  : point
        }
        guess = (short_dist_tol, point[0], point[1], point[2])
        short_dist, new_point[0], new_point[1], new_point[2] = fsolve(solve_eq, guess, args=params)
        # new_point = P(short_dist, point, period=period)

        # solve for the distance from the new point to another point on the minimal surface
        params = {
            'period' : period,
            'point'  : new_point
        }
        guess = (guess_dist, new_point[0], new_point[1], new_point[2])
        sol = fsolve(solve_eq, guess, args=params)

        distribution[i] = sol[0]

    
    return distribution



# Some hard coded parameters to use in all distributions
n = 100                 # grid size for discretized structure
box = 12.2               # box size in nm
period = box            # period of the minimal surface in nm
struct_tol = 0.01       # tolerance for locating discretized points on the surface
guess = 4.6             # initial guess for the numerical solvers --> corresponds to the bilayer distance + expected pore size from MD and experiment
sample = 1000

gyroid = True
scharzD = False
primitive = False

if gyroid:

    # Generate the structure
    x = np.linspace(0,    box, n)
    y = np.linspace(0,    box, n)
    z = np.linspace(0,    box, n)
    X = [x[:,None,None], y[None,:,None], z[None,None,:]]

    C = Gyroid(X, period)

    grid = np.zeros([n**3, 3])
    count = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if -struct_tol < C[i,j,k] < struct_tol:
                    grid[count,:] = [x[i], y[j], z[k]]
                    count += 1
                    
    structure = grid[:count, :]

    # Generate the distribution
    gyroid_dist = surface2surface(structure,struct='gyroid',box=box,period=period,sample=sample)
    plt.hist(gyroid_dist)
    plt.show()