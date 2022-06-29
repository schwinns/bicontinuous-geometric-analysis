# Script to plot pore-to-pore distance distributions from txt data files

import numpy as np
import matplotlib.pyplot as plt

# filename = 'pore2pore'
# gyroid = np.loadtxt(filename + '_gyroid.txt')
# schwarz = np.loadtxt(filename + '_schwarz.txt')
# primitive = np.loadtxt(filename + '_primitive.txt')

gyroid = np.loadtxt('data/gyroid_with_error.txt')
schwarz = np.loadtxt('data/schwarz_with_error.txt')
primitive = np.loadtxt('data/primitive_with_error.txt')

fig, ax = plt.subplots(1,1, figsize=(10,8))

# Plot the KDE plots for each space group
ax.plot(gyroid[:,0], gyroid[:,1], c='tab:blue', label='Gyroid')
ax.fill_between(gyroid[:,0], gyroid[:,2], gyroid[:,3], color='tab:blue', alpha=0.25)

ax.plot(schwarz[:,0], schwarz[:,1], c='tab:orange', label='Schwarz Diamond')
ax.fill_between(schwarz[:,0], schwarz[:,2], schwarz[:,3], color='tab:orange', alpha=0.25)

ax.plot(primitive[:,0], primitive[:,1], c='tab:green', label='Primitive')
ax.fill_between(primitive[:,0], primitive[:,2], primitive[:,3], color='tab:green', alpha=0.25)

# Add the bilayer thickness + pore size line
ax.axvline(4.6, color='black', linestyle='dashed', label='Expected pore-to-pore distance')
ax.axvspan(4.6 - 0.25, 4.6 + 0.25, color='gray', alpha=0.5)

ax.axvline(4.6*2, color='black', linestyle='dashed')
ax.axvspan(4.6*2 - 0.25, 4.6*2 + 0.25, color='gray', alpha=0.5)

# Some formatting
ax.set_xlim(3,10)
ax.set_xlabel('distance (nm)',fontsize='large')
ax.set_ylabel('probability density',fontsize='large')
ax.legend(fontsize='x-large',loc=1)
plt.show()