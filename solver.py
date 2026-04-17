#! /bin/python3

## Contains Functions for Solving the Time Dependent Linear Schrodinger Equation in 1D
#  Using a Finite Difference Method

# Load modules used
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags

## Configuration

# Set up x grid
x0 = 0                            # x grid lower bound
xf = 1                            # x grid upper bound
dx = 0.01                         # x grid spacing
xNum = int( ( xf - x0 ) / dx)     # number of points in the x grid
x = np.linspace( x0, xf, xNum )   # define x grid

# Set up t grid
t0 = 0                            # start time
tf = 1                            # final time
dt = 0.01                         # time grid spacing
tNum = int( ( tf - t0 ) / dt )    # number of points in the t grid
t = np.linspace( t0, tf, tNum )   # define t grid

alpha = dx**2/dt                  # ensure alpha is <1 to maintain stability

#print("alpha equals {}".format(alpha))          # test alpha


# Define seed (soultion to the infinite square well in this case)
seed = np.zeros(len(x))
#print("seed initialization: {}".format(seed))   # test seed init

seed = np.sqrt(2) *np.sin(np.pi * x )
#print("seed initialization: {}".format(seed))   # test seed construction

# Plot Seed
plt.plot(x,seed,'*')
plt.xlabel( "x(0)" )
plt.ylabel( "Psi(x)" )
plt.title(  "Infinite Square Well Seed Wavefunction" )
plt.savefig('figures/ISW_seed.png')


## Construct Matrix to calculate psi(t+1)


def create_tridiagonal(n, a, b, c):
    '''The function `create_tridiagonal(n, a, b, c)
       constructs an n x n matrix, setting values along the main diagonal (b),
       the upper diagonal (a), and the lower diagonal (c) '''
    # np.fill_diagonal does not work for complex numbers
    # use scipy.spares.diags
    diagonal  = [b] * n
    Udiagonal = [a] * ( n-1 )
    Ldiagonal = [c] * ( n-1 )
    matrix = diags( [ Udiagonal, diagonal, Ldiagonal ], [ 1, 0, -1], shape=(n,n)).toarray()
    return matrix

# Define the matrix elements
aD  = 1 - (1j/alpha)                # diagonal element
aUD = (1j/2)*(1/alpha)              # upper diagonal element
aLD = (1j/2)*(1/alpha)              # lower diagonal element

# Construct the matrix using function above
triDiagMat = create_tridiagonal( len(x), aUD, aD, aLD)
#print( triDiagMat)  #test triDiagMat constructor function

# Calculate the wavefunction at the first timestep (psi_1 )
psi_1 = np.zeros(len(x))           # initialize psi_1

# Perform calculation of wavefunction at first time step
psi_1 = triDiagMat @ seed

# Plot psi_1
plt.plot(x,seed,'*')
plt.xlabel( "x(1)" )
plt.ylabel( "Psi(x)" )
plt.title(  "Infinite Square Well Wavefunction at Fist Time Step" )
plt.savefig('figures/ISW_psi_1.png')
