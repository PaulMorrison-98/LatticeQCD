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
N=16
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
########################################################################

#Returns conjugate transpose of a complex matrix
########################################################################
def conj(matrix):
    return matrix.conj().T

#Returns the modulus of argument
def mod(x, n=N):
    return x % n

#Returns the product of matrices
def matmul(*matrices):
    product = I
    for i in matrices:
        product = np.dot(product, i)
    return product
########################################################################

#SU(3) matrix Generator
########################################################################
def matrix_su2(epsilon = 0.2):
    ### Pauli matrices
    sigma1 = np.array([[0, 1], [1, 0]])
    sigma2 = np.array([[0, -1J], [1J, 0]])
    sigma3 = np.array([[1, 0], [0, -1]])
    r = [0., 0., 0., 0.]
    for i in range(4):
        r[i] = (np.random.uniform(0, 0.5))
    ### normalize
    norm = np.sqrt(r[1]**2 + r[2]**2 + r[3]**2)
    r[1:] = map(lambda x: epsilon*x / norm, r[1:])
    r[0]  = np.sign(r[0]) * np.sqrt(1. - epsilon**2)
    M = np.identity(2, dtype='complex128')
    M = M * r[0]
    M = np.add(1J * r[1] * sigma1, M)
    M = np.add(1J * r[2] * sigma2, M)
    M = np.add(1J * r[3] * sigma3, M)
    return M

#Generates Need 3 SU(3) matrices for one SU(3) matrix
#From Gattringer & Lang's textbook.
def matrix_su3(epsilon = 0.2):
    R_su2 = matrix_su2(epsilon)
    S_su2 = matrix_su2(epsilon)
    T_su2 = matrix_su2(epsilon)
    # initialise to identity, need complex numbers from now
    R = np.identity(3, dtype='complex128')
    S = np.identity(3, dtype='complex128')
    T = np.identity(3, dtype='complex128')
    # upper
    R[:2,:2] = R_su2
    # edges
    S[0:3:2, 0:3:2] = S_su2
    # lower
    T[1:,1:] = T_su2
    # create final matrix
    X = np.dot(R, S)
    return np.dot(X, T)
    
#Create set of SU(3) matrices
#Needs to be large enough to cover SU(3)
def generate_SU3(epsilon = esp, tot = 1000):
    matrices = []
    for i in range(tot):
        X = matrix_su3(epsilon)
        matrices.append(X)
        matrices.append(X.conj().T)
    return matrices
########################################################################

#Calculates and returns the change in the action
########################################################################
    M = SU3_bank[np.random.randint(101)]
    delta_s = (-beta/3)*(np.trace(np.real(matmul((matmul(M,U) -U),staple))))
    return delta_s, M
########################################################################

#Performs update sweep
########################################################################
def update(L,SU3_bank):
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
    for p in product(range(N), range(N), range(N), range(N), range(4)):
        x, y, z, t, n = p
        staple = np.zeros((3, 3), dtype = complex)
        #This denotes the particular direction for this iteration (the convention that I have adopted is 0=x-direction, 1=y-direction, 2=z-direction, 3=z-direction, 4=t-direction). 
        #It is used similarly to xc, yc, ect. when n  in  equals 0 only U_dirc[0] will equal 1 and all others will be zero.
        U_dirc = [0] * 4
        U_dirc[n] = 1
        U = L[x][y][z][t][n]
        #calculate positive staples (these are the staples for whom the first link starts in the positive direction)
        for i in range(4):
            if i ==n:
                continue
            staple +=matmul(L[mod(x +U_dirc[0])][mod(y +U_dirc[1])][mod(z +U_dirc[2])][mod(t +U_dirc[3])][i],
                                  conj(L[mod(x +xc[i])][mod(y + yc[i])][mod(z +zc[i])][mod(t +tc[i])][n]),
                                  conj(L[mod(x)][mod(y)][mod(z)][mod(t)][i]))
            
        #calculate negative staples  (these are the staples for whom the first link starts in the negative direction)
        for i in range(4):
            if i ==n:
                continue
            staple += matmul(conj(L[mod(x -xc[i] +U_dirc[0])][mod(y -yc[i] +U_dirc[1])][mod(z -zc[i] +U_dirc[2])][mod(t -tc[i] +U_dirc[3])][i]),
                                  conj(L[mod(x -xc[i])][mod(y - yc[i])][mod(z -zc[i])][mod(t -tc[i])][n]),
                                  L[mod(x-xc[i])][mod(y-yc[i])][mod(z-zc[i])][mod(t-tc[i])][i])
            
        #Then the staples for a particular link are summed the change in a random SU(3) matrix, M, is multipled with the link in question, U, and
        # the change in action is calculated to determine if the aletered link should be kept
        for p in range(N_slj):
            delta_S , M = ds(U,staple, SU3_bank)
            if delta_S < 0 or np.random.random() < np.exp(-delta_S):
                L[x][y][z][t][n] = matmul(M, L[x][y][z][t][n])
########################################################################

#Measures the average plaquette
########################################################################  
def measure_plaq(L):
    xc = [0] * 4
    yc = [0] * 4
    zc = [0] * 4
    tc = [0] * 4
    xc[0] = 1
    yc[1] = 1
    zc[2] = 1
    tc[3] = 1
    plaq_ave = 0
    plaq_sum = 0

    for mu in range(1, 4):
        for nu in range(mu):
            for p in product(range(N), range(N), range(N), range(N)):
                x, y, z, t = p
                plaq_sum += np.trace(np.real(matmul(L[mod(x)][mod(y)][mod(z)][mod(t)][nu],
                                    L[mod(x +xc[nu])][mod(y + yc[nu])][mod(z +zc[nu])][mod(t +tc[nu])][mu],
                                    conj(L[mod(x +xc[mu])][mod(y + yc[mu])][mod(z +zc[mu])][mod(t +tc[mu])][nu]),
                                    conj(L[mod(x)][mod(y)][mod(z)][mod(t)][mu]))))
    plaq_ave = (plaq_sum)/ N**4 / 18
    return plaq_ave
########################################################################

#Useful function for writing results to file
########################################################################
def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)
########################################################################

#Main
######################################################################## 
# Generrate bank of SU3 matrices    
SU3_bank = generate_SU3()
#Intialising the lattice.
#At each point in the 4-D hyper-lattice there are 4 possible directions (if we, to avoid double counting, restrict ourselves to only the positive directions)
#you we can move. As each link is represnted by a SU(3) mattrix there will be a 3x3 matrix associated to each one of the 4(N^4) lattice sites.
L = np.array([I] * N**4 * 4).reshape(N,N,N,N,4,3,3)


for i in range(1600):
    start_time = time.time()
    for dumb in range(N_cor):
        update(L, SU3_bank)
    
    end_time = time.time() - start_time
    print(end_time)

    plaq_ave = measure_plaq(L)
    print(plaq_ave)
    row_contents = [plaq_ave, run_numb]
    Append a list as new line to an old csv file
    append_list_as_row('text.csv', row_contents) 