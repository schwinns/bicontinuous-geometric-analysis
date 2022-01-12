#!/usr/bin/env python

import argparse
import numpy as np
import plotly.express as px
import pandas as pd
import random

parser = argparse.ArgumentParser()
parser.add_argument('-s','--structure',default='gyroid',
                    help='BCC structure to generate surface-to-surface distances')
parser.add_argument('-b','--box',default=9.4,type=float,
                    help='size of the box (nm)')
parser.add_argument('-p','--period',default=9.4,type=float,
                    help='period for the triply periodic minimal surface')
parser.add_argument('-n','--n',default=100,type=int,
                    help='grid size for generating the surface')
parser.add_argument('-i','--increment',default=0.05,type=float,
                    help='increment to extend along the normal on each iteration')
parser.add_argument('-t','--tolerance',default=0.1,type=float,
                    help='tolerance for determining when increment point equals a surface point')
parser.add_argument('-sample','--sample',default=100,type=int,
                    help='sample size for the distribution of surface-to-surface distances')
parser.add_argument('-o','--output',default='histogram',
                    help='output gro filename')
args = parser.parse_args()

# Some convenience functions
def sin(x):
    return np.sin(x)

def cos(x):
    return np.cos(x)

def sqrt(x):
    return np.sqrt(x)

pi = np.pi

# Define the triply periodic minimal surface functions
def SchwarzD(X, period):

    N = 2*pi/period
    
    a = sin(N*X[0]) * sin(N*X[1]) * sin(N*X[2])
    b = sin(N*X[0]) * cos(N*X[1]) * cos(N*X[2])
    c = cos(N*X[0]) * sin(N*X[1]) * cos(N*X[2])
    d = cos(N*X[0]) * cos(N*X[1]) * sin(N*X[2])
    
    return a + b + c + d

def Gyroid(X,period):
    
    N = 2*pi/period
    
    a = sin(N*X[0]) * cos(N*X[1])
    b = sin(N*X[1]) * cos(N*X[2])
    c = sin(N*X[2]) * cos(N*X[0])
    
    return a + b + c

def SchwarzD_grad(v,period):
    
    x = v[0]; y = v[1]; z = v[2]
    N = 2*pi / period
    
    a = N*cos(N*x)*sin(N*y)*sin(N*z) + N*cos(N*x)*cos(N*y)*cos(N*z) - N*sin(N*x)*sin(N*y)*cos(N*z) - N*sin(N*x)*cos(N*y)*sin(N*z)
    b = N*sin(N*x)*cos(N*y)*sin(N*z) - N*sin(N*x)*sin(N*y)*cos(N*z) + N*cos(N*x)*cos(N*y)*cos(N*z) - N*cos(N*x)*sin(N*y)*sin(N*z)
    c = N*sin(N*x)*sin(N*y)*cos(N*z) - N*sin(N*x)*cos(N*y)*sin(N*z) - N*cos(N*x)*sin(N*y)*sin(N*z) + N*cos(N*x)*cos(N*y)*cos(N*z)
    
    return np.array([a,b,c]) / np.linalg.norm(np.array([a,b,c]))

def Gyroid_grad(v,period):
    
    x = v[0]; y = v[1]; z = v[2]
    N = 2*pi / period
    
    a =  N*cos(N*x)*cos(N*y) - N*sin(N*x)*sin(N*z)
    b = -N*sin(N*y)*sin(N*x) + N*cos(N*y)*cos(N*z)
    c = -N*sin(N*y)*sin(N*z) + N*cos(N*z)*cos(N*x)
    
    return np.array([a,b,c]) / np.linalg.norm(np.array([a,b,c]))

def surface2surface(structure,struct='gyroid',box=9.4,period=9.4,sample=100,increment=0.05,tol=0.1,verbose=False):
    distribution = []
    count = 0
    while len(distribution) < sample:

        p = random.randint(0, structure.shape[0] - 1)
        if verbose:
            print('point number: %d' %(p))
        point = structure[p,:]
        if struct == 'gyroid':
            norm = Gyroid_grad(point,period)
        elif struct == 'schwarzD':
            norm = SchwarzD_grad(point,period)

        new_point = point + norm*increment # get first incremented point
        i = 1 # track number of increments
        check = False
        while not check:
    
            for s,struct in enumerate(structure): # search whole structure
                if not s == p: # do not check when original random point
                    d1 = new_point - struct # calculate distance between each point on structure and the incremented point
                    d2 = np.sqrt(d1[0]**2 + d1[1]**2 + d1[2]**2)
                    if d2 < tol: # if incremented point = point on structure
                        if verbose:
                            print('struct: %s' %(struct))
                            print('new_point: %s' %(new_point))
                            print('how equal: %s' %(d2))
                        distribution.append(i*increment) # add the distance incremented to the distribution (aka the distance between points on structure along normal)
                        check = True
                        count += 1
                        print(i*increment)
                        print('Saved %d out of %d distances to distribution' %(count,sample))
                        break
        
            new_point = new_point + norm*increment
            i += 1
            if new_point[0] < 0 or new_point[1] < 0 or new_point[2] < 0:
                if verbose:
                    print('Incrementing point has left the box. Selecting new point.')
                break
            if new_point[0] > box or new_point[1] > box or new_point[2] > box:
                if verbose:
                    print('Incrementing point has left the box. Selecting new point.')
                break
    
    return distribution

n = args.n

x = np.linspace(0, args.box, n)
y = np.linspace(0, args.box, n)
z = np.linspace(0, args.box, n)
X = [x[:,None,None], y[None,:,None], z[None,None,:]]

if args.structure == 'gyroid':
    C = Gyroid(X, args.period)
elif args.structure == 'schwarzD':
    C = SchwarzD(X, args.period)
else:
    raise ValueError('No valid structure provided. Please choose either gyroid or schwarzD')

grid = np.zeros([n**3, 3])
count = 0
for i in range(n):
    for j in range(n):
        for k in range(n):
            if -0.01 < C[i,j,k] < 0.01:
                grid[count,:] = [x[i], y[j], z[k]]
                count += 1
                
struct = grid[:count, :]

dist = surface2surface(struct,struct=args.structure,box=args.box,
                       period=args.period,sample=args.sample,
                       increment=args.increment,tol=args.tolerance)

out = open(args.output + '.data', 'w')
for l in dist:
	out.write(str(l) + '\n')

fig.show()

#fig.write_image(args.output + '.png')
