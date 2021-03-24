#Imports and constant parameters
####################################################################
import numpy as np
import cmath
import time
import sys
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

#Mean calculations
#######################################################################
sum_of_action_density = 0 
sum_of_topo_density = 0
#sum_of_action_density = [[[[0 for x in range(8)] for y in range(8)] for z in range(8)] for t in range(8)]
for idx in range(150,1599):
    input_idx = '/home/paulmorrison/Documents/QCD_project/action_density/configurations/action_density_cfg' + str(int( idx ))
    tmp = np.load(input_idx)
    density = [[[[0 for x in range(8)] for y in range(8)] for z in range(8)] for t in range(8)]
    
    for x in range(8):
        for y in range(8):
            for z in range(8):
                for t in range(8):
                    density[x][y][z][t] = tmp[x][y][z][t]
    
    sum_of_action_density =  np.add(sum_of_action_density, density)
    

for idx in range(150,1599):
    input_idx = '/home/paulmorrison/Documents/QCD_project/Topological_charge_density/Topological_charge_cfg' + str(int( idx ))
    tmp = np.load(input_idx)
    density = [[[[0 for x in range(8)] for y in range(8)] for z in range(8)] for t in range(8)]
    
    for x in range(8):
        for y in range(8):
            for z in range(8):
                for t in range(8):
                    density[x][y][z][t] = tmp[x][y][z][t]
    
    
    sum_of_topo_density =  np.add(sum_of_topo_density, density)    

MC_density_action= np.true_divide(sum_of_action_density, 1448)
MC_density_topo= np.true_divide(sum_of_topo_density, 1448)
full_mean_action = np.sum(MC_density_action)
full_mean_topo = np.sum(MC_density_topo)
#######################################################################

#Coeffienct calculation (A= action density, B = Topo_density)
 #######################################################################
topo_charge = 0
DENOMINATOR_term1 = 0 
DENOMINATOR_term2 = 0
NUMERATOR = 0
for idx in range(150,1599):
    sum_term = 0
    input_idx = '/home/paulmorrison/Documents/QCD_project/action_density/configurations/action_density_cfg' + str(int( idx ))
    tmp = np.load(input_idx)
    a = [[[[0 for x in range(8)] for y in range(8)] for z in range(8)] for t in range(8)]
    b = [[[[0 for x in range(8)] for y in range(8)] for z in range(8)] for t in range(8)]

    for x in range(8):
        for y in range(8):
            for z in range(8):
                for t in range(8):
                    a[x][y][z][t] = tmp[x][y][z][t]

    input_idx = '/home/paulmorrison/Documents/QCD_project/Topological_charge_density/Topological_charge_cfg' + str(int( idx ))
    tmp = np.load(input_idx)

    for x in range(8):
        for y in range(8):
            for z in range(8):
                for t in range(8):
                    b[x][y][z][t] = tmp[x][y][z][t]

    A = np.asarray(a)
    B = np.asarray(b)

    action_element = np.sum(A)
    topo_element = np.sum(B)

    NUMERATOR =  NUMERATOR + (action_element - full_mean_action)*(topo_element - full_mean_topo)
    DENOMINATOR_term1 = DENOMINATOR_term1 + (action_element - full_mean_action)**2
    DENOMINATOR_term2 = DENOMINATOR_term2 + (topo_element - full_mean_topo)**2


Rab = (NUMERATOR)/(np.sqrt((DENOMINATOR_term1)*(DENOMINATOR_term2)))
print(Rab)
