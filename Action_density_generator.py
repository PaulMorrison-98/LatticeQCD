#Imports and constant parameters
####################################################################
import numpy as np
import cmath
import time
from itertools import product
i= cmath.sqrt(-1)
I = np.identity(3,dtype=complex)
from csv import writer
# Lattice size
N=8
# Number of passes before evaulation of the action
N_cor = 50
# Number of configurations 
N_conf = 25
# number of "single lattice jiggles" (slj)
N_slj = 10
# Prefactor as suggested by Lepage (to be altered)
beta = 5.5
# RNG distribution range
esp=0.24
#######################################################################

#Useful functions
#######################################################################
#Returns conjugate transpose of a complex matrix
def conj(matrix):
    return matrix.conj().T

#Returns the modulus of argument
def mod(x, n=N):
    return x % n

#Returns the 
def matmul(*matrices):
    product = I
    for i in matrices:
        product = np.dot(product, i)
    return product
#######################################################################

#Generates the action density for a given configuration 
#######################################################################
def local_action(L):
    #These variables are used to move through the lattice. e.g. when i in the following loop equals 1 only yc will equal 1 and all others will be zero.
    xc = [0] * 4
    yc = [0] * 4
    zc = [0] * 4
    tc = [0] * 4
    xc[0] = 1
    yc[1] = 1
    zc[2] = 1
    tc[3] = 1
    #loops through the x,y,z,t lattice sites and the 4 postive directions associated to each one
    for p in product(range(N), range(N), range(N), range(N)):
        x, y, z, t = p
        local_action = 0
        plaqs = 0
        for n in range(4):
            #This denotes the particular direction for this iteration (the convention that I have adopted is 0=x-direction, 1=y-direction, 2=z-direction, 3=z-direction, 4=t-direction). 
            #It is used similarly to xc, yc, ect. when n  in  equals 0 only U_dirc[0] will equal 1 and all others will be zero.
            U_dirc = [0] * 4
            U_dirc[n] = 1
            U = L[x][y][z][t][n]
            #calculate positive staples (these are the staples for whom the first link starts in the positive direction)
            for i in range(4):
                if i ==n:
                    continue
                plaqs += 1 - (1/3)*(np.trace(np.real((matmul(U,matmul(L[mod(x +U_dirc[0])][mod(y +U_dirc[1])][mod(z +U_dirc[2])][mod(t +U_dirc[3])][i],
                                    conj(L[mod(x +xc[i])][mod(y + yc[i])][mod(z +zc[i])][mod(t +tc[i])][n]),
                                    conj(L[mod(x)][mod(y)][mod(z)][mod(t)][i])))))))
                
            #calculate negative staples  (these are the staples for whom the first link starts in the negative direction)
            for i in range(4):
                if i ==n:
                    continue
                plaqs += 1 - (1/3)*(np.trace(np.real((matmul(U,matmul(conj(L[mod(x -xc[i] +U_dirc[0])][mod(y -yc[i] +U_dirc[1])][mod(z -zc[i] +U_dirc[2])][mod(t -tc[i] +U_dirc[3])][i]),
                                    conj(L[mod(x -xc[i])][mod(y - yc[i])][mod(z -zc[i])][mod(t -tc[i])][n]),
                                    L[mod(x-xc[i])][mod(y-yc[i])][mod(z-zc[i])][mod(t-tc[i])][i]))))))
 
        local_action = (beta)*(plaqs)
        action_density[x][y][z][t] = local_action

    return action_density 
#######################################################################

#Main
#######################################################################
sum_of_density =0 
action_density = np.zeros((8, 8, 8, 8))
for idx in range(150,1599):
    input_idx = '/home/paulmorrison/Documents/QCD_project/configurations/cfg' + str(int( idx ))
    tmp = np.load(input_idx)
    L = [[[[[0 for mu in range(4)] for t in range(8)] for z in range(8)] for y in range(8)] for x in range(8)]
        
    for x in range(8):
        for y in range(8):
            for z in range(8):
                for t in range(8):
                    for mu in range(4):
                        L[x][y][z][t][mu] = np.matrix(tmp[x][y][z][t][mu])
    
    element = local_action(L)
    output_idx = 'action_density_cfg' + str(int( idx ))
    file_out = open(output_idx, 'wb')
    np.save(file_out, element)