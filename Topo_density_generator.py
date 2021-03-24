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

def fn_topological_charge(U, x, y, z, t):
    F01 = fn_F_munu(U, x, y, z, t, 0, 1)
    F23 = fn_F_munu(U, x, y, z, t, 2, 3)
    F02 = fn_F_munu(U, x, y, z, t, 0, 2)
    F31 = fn_F_munu(U, x, y, z, t, 3, 1)
    F03 = fn_F_munu(U, x, y, z, t, 0, 3)
    F12 = fn_F_munu(U, x, y, z, t, 1, 2)
    result = np.trace( np.dot(F01, F23) + np.dot(F02, F31) + np.dot(F03, F12))
    return result / ( 4. * np.pi**2 )

### antihermitian, traceless version of field strength
def fn_F_munu(U, x, y, z, t, mu, nu):
    Pmunu = fn_plaquette(U, x, y, z, t, mu, nu)
    return -1.0J * (np.subtract(Pmunu, Pmunu.conj().T) - np.trace(np.subtract(Pmunu, Pmunu.conj().T)) / 3.) / 2.

def fn_plaquette(U, x, y, z, t, mu, nu):
    Nt = len(U)
    Nx = len(U[0])
    Ny = len(U[0][0])
    Nz = len(U[0][0][0])
    start_txyz = [x,y,z,t]
    result = 1.
    result, next_txyz = fn_line_move_forward(U, 1., start_txyz, mu)
    result, next_txyz = fn_line_move_forward(U, result, next_txyz, nu)
    result, next_txyz = fn_line_move_backward(U, result, next_txyz, mu)
    result, next_txyz = fn_line_move_backward(U, result, next_txyz, nu)    
    return result

def fn_periodic_link(U, txyz, direction):
    Nx, Ny, Nz, Nt = len(U), len(U[0]), len(U[0][0]), len(U[0][0][0])
    return U[txyz[0] % Nx][txyz[1] % Ny][txyz[2] %Nz][txyz[3] % Nt][direction]

def fn_move_forward_link(U, txyz, direction):
    link = fn_periodic_link(U, txyz, direction)
    new_txyz = txyz[:]
    new_txyz[direction] += 1
    return link, new_txyz

def fn_move_backward_link(U, txyz, direction):
    new_txyz = txyz[:]
    new_txyz[direction] -= 1
    link = fn_periodic_link(U, new_txyz, direction).conj().T
    return link, new_txyz

def fn_line_move_forward(U, line, txyz, direction):
    link, new_txyz = fn_move_forward_link(U, txyz, direction)
    new_line = np.dot(line, link)
    return new_line, new_txyz

def fn_line_move_backward(U, line, txyz, direction):
    link, new_txyz = fn_move_backward_link(U, txyz, direction)
    new_line = np.dot(line, link)
    return new_line, new_txyz
for idx in range(150,1599):
    topo_density = np.zeros((8, 8, 8, 8))
    input_idx = '/home/paulmorrison/Documents/QCD_project/configurations/cfg' + str(int( idx ))
    tmp = np.load(input_idx)
    L = [[[[[0 for mu in range(4)] for z in range(8)] for y in range(8)] for x in range(8)] for t in range(8)]
        
    for t in range(8):
        for x in range(8):
            for y in range(8):
                for z in range(8):
                    for mu in range(4):
                        L[t][x][y][z][mu] = np.matrix(tmp[t][x][y][z][mu])

    for p in product(range(N), range(N), range(N), range(N)):
        x, y, z, t = p

        topo_density[x][y][z][t] = fn_topological_charge(L,x, y, z, t).real
        
    output_idx = 'Topological_charge_cfg' + str(int( idx ))
    file_out = open(output_idx, 'wb')
    np.save(file_out, topo_density)

