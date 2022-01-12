#!/usr/bin/env python

# Import packages
import numpy as np
import pandas as pd
import random
from scipy.optimize import fsolve
import scipy.optimize
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s','--struct',nargs='+',
                    help='structure to generate distribution for')
args = parser.parse_args()


# Define the triply periodic minimal surface functions
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


# Functions for solving the equations

def P(t,X,period=9.4):
    n = Gyroid_grad(X,period)
    
    return X + t*n

def Gyroid_eq(X,period=9.4):

    n = Gyroid_grad(X,period)
    N = 2*np.pi/period
    
    return lambda t : np.sin(N * (X[0] + t*n[0])) * np.cos(N * (X[1] + t*n[1])) + np.sin(N * (X[1] + t*n[1])) * np.cos(N * (X[2] + t*n[2])) + np.sin(N * (X[2] + t*n[2])) * np.cos(N * (X[0] + t*n[0]))

def SchwarzD_eq(X,period=9.4):
    
    n = SchwarzD_grad(X,period)
    N = 2*np.pi/period
    
    return lambda t : np.sin(N * (X[0] + t*n[0])) * np.sin(N * (X[1] + t*n[1])) * np.sin(N * (X[2] + t*n[2])) + np.sin(N * (X[0] + t*n[0])) * np.cos(N * (X[1] + t*n[1])) * np.cos(N * (X[2] + t*n[2])) + np.cos(N * (X[0] + t*n[0])) * np.sin(N * (X[1] + t*n[1])) * np.cos(N * (X[2] + t*n[2])) + np.cos(N * (X[0] + t*n[0])) * np.cos(N * (X[1] + t*n[1])) * np.sin(N * (X[2] + t*n[2]))

def Primitive_eq(X,period=9.4):

    n = Primitive_grad(X,period)
    N = 2*np.pi/period
    
    return lambda t : np.cos(N * (X[0] + t*n[0])) + np.cos(N * (X[1] + t*n[1])) + np.cos(N * (X[2] + t*n[2]))


# Main function for generating the distribution
def surface2surface(structure,struct='gyroid',guess=4.5,box=9.4,period=9.4,sample=10):
    distribution = []
    ps = []
    structs = np.zeros([sample,3])
    count = 0
    while len(distribution) < sample:

        p = random.randint(0, structure.shape[0] - 1)
        point = structure[p,:]
        if struct == 'gyroid':
            n = Gyroid_grad(point,period)
            sol = fsolve(Gyroid_eq(point,period), guess)
        elif struct == 'schwarzD':
            n = SchwarzD_grad(point,period)
            sol = fsolve(SchwarzD_eq(point,period), guess)
        elif struct == 'primitive':
            n = Primitive_grad(point,period)
            sol = fsolve(Primitive_eq(point,period), guess)
        
        if not sol > box and sol > 0:
            distribution.append(sol)
            structs[count,:] = P(sol,point,period=period)
            ps.append(p)
            count += 1

            if sol < 0.1:
                print(sol)
                
    return distribution, structs, ps

# Solve using different solvers instead of fsolve (Newton)
def surface2surface_test(structure,struct='gyroid',a=0,b=10,box=9.4,period=9.4,sample=10):
    distribution = []
    ps = []
    structs = np.zeros([sample,3])
    count = 0
    while len(distribution) < sample:

        p = random.randint(0, structure.shape[0] - 1)
        point = structure[p,:]
        if struct == 'gyroid':
            n = Gyroid_grad(point,period)
            sol = scipy.optimize.toms748(Gyroid_eq(point,period), a,b)
        elif struct == 'schwarzD':
            n = SchwarzD_grad(point,period)
            sol = scipy.optimize.toms748(SchwarzD_eq(point,period), a,b)
        
        if not sol > box:
            distribution.append(sol)
            structs[count,:] = P(sol,point,period=period)
            ps.append(p)
            count += 1
                
    return distribution, structs, ps

# Parse arguments for which distributions to generate
# if 'gyroid' in args.struct:
#     gyroid = True
# else:
#     gyroid = False

# if 'schwarz' in args.struct:
#     schwarz = True
# else:
#     schwarz = False

# if 'primitive' in args.struct:
#     primitive = True
# else:
#     primitive = False
gyroid = False
schwarz = True
primitive = False


# Generate distributions
if gyroid:

    # ### Gyroid
    # Generate the structure
    n = 100
    box = 9.4 # nm
    period = 9.4
    x = np.linspace(0,    box, n)
    y = np.linspace(0,    box, n)
    z = np.linspace(0,    box, n)
    X = [x[:,None,None], y[None,:,None], z[None,None,:]]

    print('Box size is\t %.4f' %(box))
    print('Period is\t %.4f' %(period))

    C = Gyroid(X, period)

    grid = np.zeros([n**3, 3])
    count = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if -0.01 < C[i,j,k] < 0.01:
                    grid[count,:] = [x[i], y[j], z[k]]
                    count += 1
                    
    structure = grid[:count, :]
    print('Size of structure:')
    print(structure.shape)

    df = pd.DataFrame(structure, columns=['x','y','z'])
    df['size'] = np.ones(structure.shape[0]) * 0.1

    # #### Generate a full distribution

    # Generate the distribution
    sample = 100
    box = 9.4
    guess = 4.6
    struct = 'gyroid'

    dist_gyroid,structs,ps = surface2surface(structure,struct=struct,guess=guess,box=box,period=period,sample=sample)
    #dist_gyroid_test,structs_test,ps_test = surface2surface_test(structure,struct=struct,a=a,b=b,box=box,period=period,sample=sample)

    hist_gyroid = pd.DataFrame(dist_gyroid,columns=['Gyroid'])


if schwarz:
    # ### SchwarzD

    # Generate the structure
    n = 100
    box = 9.4 # nm
    period = 9.4
    x = np.linspace(0,    box, n)
    y = np.linspace(0,    box, n)
    z = np.linspace(0,    box, n)
    X = [x[:,None,None], y[None,:,None], z[None,None,:]]

    print('Box size is\t %.4f' %(box))
    print('Period is\t %.4f' %(period))

    C = SchwarzD(X,period)

    grid = np.zeros([n**3, 3])
    count = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if -0.01 < C[i,j,k] < 0.01:
                    grid[count,:] = [x[i], y[j], z[k]]
                    count += 1
                    
    structure = grid[:count, :]
    print('Size of structure:')
    print(structure.shape)

    df = pd.DataFrame(structure, columns=['x','y','z'])
    df['size'] = np.ones(structure.shape[0]) * 0.1

    # #### Generate a full distribution 

    # Generate the distribution
    sample = 100
    box = 9.4
    guess = 4.6
    struct = 'schwarzD'

    dist_schwarzD,structs,ps = surface2surface(structure,struct=struct,guess=guess,box=box,period=period,sample=sample)
    hist_schwarzD = pd.DataFrame(dist_schwarzD,columns=['SchwarzD'])

if primitive:
    # ## Primitive
    # Generate the structure
    n = 100
    box = 9.4 # nm
    period = 9.4
    x = np.linspace(0,    box, n)
    y = np.linspace(0,    box, n)
    z = np.linspace(0,    box, n)
    X = [x[:,None,None], y[None,:,None], z[None,None,:]]

    print('Box size is\t %.4f' %(box))
    print('Period is\t %.4f' %(period))

    C = Primitive(X, period)

    grid = np.zeros([n**3, 3])
    count = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if -0.01 < C[i,j,k] < 0.01:
                    grid[count,:] = [x[i], y[j], z[k]]
                    count += 1
                    
    structure = grid[:count, :]
    print('Size of structure:')
    print(structure.shape)

    df = pd.DataFrame(structure, columns=['x','y','z'])
    df['size'] = np.ones(structure.shape[0]) * 0.1

    # Generate the distribution
    sample = 100
    box = 9.4
    guess = 4.6
    struct = 'primitive'

    dist_primitive,structs,ps = surface2surface(structure,struct=struct,guess=guess,box=box,period=period,sample=sample)
    hist_primitive = pd.DataFrame(dist_primitive,columns=['Primitive'])




# Plot histograms with matplotlib
bins = np.linspace(0,10,50)

fig, ax = plt.subplots(1,1, figsize=(10,8))
if gyroid:
    ax.hist(hist_gyroid['Gyroid'], bins=bins,
            alpha=0.5, label='Gyroid')
if schwarz:
    ax.hist(hist_schwarzD['SchwarzD'], bins=bins,
            alpha=0.5, label='Schwarz Diamond')
if primitive:
    ax.hist(hist_primitive['Primitive'], bins=bins,
           alpha=0.5, label='Primitive')


# Add the bilayer thickness + pore size line
x = np.ones(10)*4.59
y = np.linspace(0,100,10)
ax.plot(x,y, color='black', linestyle='dashed',label='Expected pore-to-pore distance')
ax.axvspan(4.59 - 0.17, 4.59 + 0.17, color='gray', alpha=0.5)


# Some formatting
ax.set_xlim(0,10)
ax.set_ylim(0,100)
ax.set_xlabel('distance (nm)',fontsize='large')
ax.set_ylabel('counts',fontsize='large')
ax.set_xticks(np.arange(0,11,1))
ax.legend(fontsize='x-large')
fig.savefig('output.png')





