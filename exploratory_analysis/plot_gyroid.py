# Script to plot the gyroid surface with matplotlib

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--period', default=9.4, type=float, help='periodicity for the gyroid')
parser.add_argument('-a', '--alpha', default=0.5, type=float, help='transparency for the gyroid surface')
args = parser.parse_args()

def plot_implicit(fn, bbox=(-2.5,2.5), a=1):
    ''' create a plot of an implicit function
    fn  ...implicit function (plot where fn==0)
    bbox ..the x,y,and z limits of plotted interval'''
    xmin, xmax, ymin, ymax, zmin, zmax = bbox*3
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    A = np.linspace(xmin, xmax, 100) # resolution of the contour
    B = np.linspace(xmin, xmax, 100) # number of slices
    A1,A2 = np.meshgrid(A,A) # grid on which the contour is plotted

    for z in B: # plot contours in the XY plane
        X,Y = A1,A2
        Z = fn(X,Y,z)
        cset = ax.contour(X, Y, Z+z, [z], zdir='z', alpha=a, colors='slategray')
        # [z] defines the only level to plot for this contour for this value of z

    for y in B: # plot contours in the XZ plane
        X,Z = A1,A2
        Y = fn(X,y,Z)
        cset = ax.contour(X, Y+y, Z, [y], zdir='y', alpha=a, colors='slategray')

    for x in B: # plot contours in the YZ plane
        Y,Z = A1,A2
        X = fn(x,Y,Z)
        cset = ax.contour(X+x, Y, Z, [x], zdir='x', alpha=a, colors='slategray')

    # must set plot limits because the contour will likely extend
    # way beyond the displayed level.  Otherwise matplotlib extends the plot limits
    # to encompass all values in the contour.
    ax.set_zlim3d(zmin,zmax)
    ax.set_xlim3d(xmin,xmax)
    ax.set_ylim3d(ymin,ymax)

    return fig, ax


def Gyroid_plot(x,y,z, period=args.period):
                                     
    N = 2*np.pi/period
    
    a = np.sin(N*x) * np.cos(N*y)
    b = np.sin(N*y) * np.cos(N*z)
    c = np.sin(N*z) * np.cos(N*x)
    
    return a + b + c


###########################################################

fig, ax = plot_implicit(Gyroid_plot, bbox=(0,args.period), a=args.alpha)
plt.show()