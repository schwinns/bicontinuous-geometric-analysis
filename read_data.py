#!/usr/bin/env python

import argparse
import plotly.express as px
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-f','--file',
                    help='data file with histogram for surface2surface distribution')
args = parser.parse_args()

f = open(args.file, 'r')

dist = []
for line in f:
    if line.startswith(tuple('01234566789')):
        d = float(line.split()[0])
        if d > 0.25:
            dist.append(d)

f.close()

hist = pd.DataFrame(dist,columns=['surface-to-surface distance (nm)'])
fig = px.histogram(hist,x='surface-to-surface distance (nm)')
fig.show()
