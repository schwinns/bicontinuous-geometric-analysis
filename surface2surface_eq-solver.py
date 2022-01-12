#!/usr/bin/env python

# Import packages
import argparse
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import random
from scipy.optimize import fsolve
import scipy.optimize
import matplotlib.pyplot as plt

# Some convenience functions
def sin(x):
    return np.sin(x)

def cos(x):
    return np.cos(x)

def sqrt(x):
    return np.sqrt(x)

# Define the triply periodic minimal surface functions
def SchwarzD(X, period):

    pi = np.pi
    N = 2*pi/period
    
    a = sin(N*X[0]) * sin(N*X[1]) * sin(N*X[2])
    b = sin(N*X[0]) * cos(N*X[1]) * cos(N*X[2])
    c = cos(N*X[0]) * sin(N*X[1]) * cos(N*X[2])
    d = cos(N*X[0]) * cos(N*X[1]) * sin(N*X[2])
    
    return a + b + c + d

def Gyroid(X,period):
    
    pi = np.pi
    N = 2*pi/period
    
    a = sin(N*X[0]) * cos(N*X[1])
    b = sin(N*X[1]) * cos(N*X[2])
    c = sin(N*X[2]) * cos(N*X[0])
    
    return a + b + c

def SchwarzD_grad(v,period):
    
    pi = np.pi
    x = v[0]; y = v[1]; z = v[2]
    N = 2*pi / period
    
    a = N*cos(N*x)*sin(N*y)*sin(N*z) + N*cos(N*x)*cos(N*y)*cos(N*z) - N*sin(N*x)*sin(N*y)*cos(N*z) - N*sin(N*x)*cos(N*y)*sin(N*z)
    b = N*sin(N*x)*cos(N*y)*sin(N*z) - N*sin(N*x)*sin(N*y)*cos(N*z) + N*cos(N*x)*cos(N*y)*cos(N*z) - N*cos(N*x)*sin(N*y)*sin(N*z)
    c = N*sin(N*x)*sin(N*y)*cos(N*z) - N*sin(N*x)*cos(N*y)*sin(N*z) - N*cos(N*x)*sin(N*y)*sin(N*z) + N*cos(N*x)*cos(N*y)*cos(N*z)
    
    return np.array([a,b,c]) / np.linalg.norm(np.array([a,b,c]))

def Gyroid_grad(v,period):
    
    pi = np.pi
    x = v[0]; y = v[1]; z = v[2]
    N = 2*pi / period
    
    a =  N*cos(N*x)*cos(N*y) - N*sin(N*x)*sin(N*z)
    b = -N*sin(N*y)*sin(N*x) + N*cos(N*y)*cos(N*z)
    c = -N*sin(N*y)*sin(N*z) + N*cos(N*z)*cos(N*x)
    
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
    
    return lambda t : sin(N * (X[0] + t*n[0])) * sin(N * (X[1] + t*n[1])) * sin(N * (X[2] + t*n[2])) + sin(N * (X[0] + t*n[0])) * cos(N * (X[1] + t*n[1])) * cos(N * (X[2] + t*n[2])) + cos(N * (X[0] + t*n[0])) * sin(N * (X[1] + t*n[1])) * cos(N * (X[2] + t*n[2])) + cos(N * (X[0] + t*n[0])) * cos(N * (X[1] + t*n[1])) * sin(N * (X[2] + t*n[2]))

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
        
        if not sol > box:
            distribution.append(sol)
            structs[count,:] = P(sol,point,period=period)
            ps.append(p)
            count += 1
                
    return distribution, structs, ps

parser = argparse.ArgumentParser()
parser.add_argument('-s','--structure',default='gyroid',
                    help='BCC structure to generate surface-to-surface distances')
parser.add_argument('-b','--box',default=9.4,type=float,
                    help='size of the box (nm)')
parser.add_argument('-p','--period',default=9.4,type=float,
                    help='period for the triply periodic minimal surface')
parser.add_argument('-n','--n',default=100,type=int,
                    help='grid size for generating the surface')
parser.add_argument('-sample','--sample',default=10000,type=int,
                    help='sample size for the distribution of surface-to-surface distances')
parser.add_argument('-g','--guess',default=4,
                    help='intial guess for Newtons method nonlinear solver')
parser.add_argument('-test','--test_guess',default=False,
                    help='run a scan of initial guesses from 0 to 10')
args = parser.parse_args()

n = args.n
box = args.box
period = args.period
sample = args.sample
guess = args.guess

x = np.linspace(0,    box, n)
y = np.linspace(0,    box, n)
z = np.linspace(0,    box, n)
X = [x[:,None,None], y[None,:,None], z[None,None,:]]

# Generate the gyroid structure
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
print('Size of gyroid structure matrix:')
print(structure.shape)

# Generate the surface-to-surface distribution for the gyroid
struct = 'gyroid'
dist_gyroid,structs,ps = surface2surface(structure,struct=struct,guess=guess,box=box,period=period,sample=sample)
hist_gyroid = pd.DataFrame(dist_gyroid,columns=['Gyroid'])

# Generate the Schwarz diamond structure
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
print('Size of Schwarz structure matrix:')
print(structure.shape)

# Generate the surface-to-surface distribution for the Schwarz diamond
struct = 'schwarzD'
dist_schwarzD,structs,ps = surface2surface(structure,struct=struct,guess=guess,box=box,period=period,sample=sample)
hist_schwarzD = pd.DataFrame(dist_schwarzD,columns=['SchwarzD'])


# Plot both histograms with matplotlib
bins = np.linspace(0,10,50)

fig, ax = plt.subplots(1,1, figsize=(10,8))
ax.hist(hist_gyroid['Gyroid'], bins=bins,
        alpha=0.5, label='Gyroid')
ax.hist(hist_schwarzD['SchwarzD'], bins=bins,
        alpha=0.5, label='Schwarz Diamond')

# Add the bilayer thickness + pore size line
x = np.ones(10)*4.59
y = np.linspace(0,35000,10)
ax.plot(x,y, color='black', linestyle='dashed',label='Expected pore-to-pore distance')
ax.axvspan(4.59 - 0.17, 4.59 + 0.17, color='gray', alpha=0.5)


# Some formatting
ax.set_xlim(0,10)
ax.set_ylim(0,32500)
ax.set_xlabel('distance (nm)',fontsize='large')
ax.set_ylabel('counts',fontsize='large')
ax.set_xticks(np.arange(0,11,1))
ax.legend(fontsize='x-large')

fig.savefig('dist_compare.png')

if args.test_guess:

    print('Testing intial guesses between 0 and 10 with a 0.5 step size...')
    hist_gyroid = pd.DataFrame()
    #hist_gyroid['total'] = np.zeros(sample)
    hist_schwarzD = pd.DataFrame()
    #hist_schwarzD['total'] = np.zeros(sample)

    bins = np.linspace(0,10,50)

    for g in np.linspace(0,10,20):
        
        fig, ax = plt.subplots(1,1, figsize=(10,8))

        dist_gyroid,structs,ps = surface2surface(structure,struct='gyroid',guess=g,box=box,period=period,sample=sample)
        hist_gyroid[g] = dist_gyroid
        #hist_gyroid['total'] += dist_gyroid
        ax.hist(hist_gyroid[g],bins=bins,alpha=0.3, label=g)

        dist_schwarzD,structs,ps = surface2surface(structure,struct='schwarzD',guess=g,box=box,period=period,sample=sample)
        hist_schwarzD[g] = dist_schwarzD
        #hist_schwarzD['total'] += dist_schwarzD
        ax.hist(hist_schwarzD[g],bins=bins,alpha=0.3, label=g)

        print('Completed distirbutions for guess = %.1f' %(g))
        fig.savefig('compare_guess'+str(g)+'.png')

    #ax.hist(hist_gyroid['total'], bins=bins,alpha=0.5, label='Gyroid')
    #ax.hist(hist_schwarzD['total'], bins=bins,alpha=0.5, label='Schwarz Diamond')