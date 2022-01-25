###### real basis sp3d5 with pritning to file#########
import matplotlib.pyplot as plt
import numpy as np
from numpy import sqrt, vdot, exp

a = 1.42

R1 = [a, 0]
R1m = [-a, 0]
R2 = [-a/2, sqrt(3)*a/2]
R2m = [a/2, -a*sqrt(3)/2]
R3 = [-a/2, -sqrt(3)*a/2]
R3m = [a/2, a*sqrt(3)/2]

"""
from Si
V_sssi = -6.769
#p
V_spsi = 5.58
V_ppsi = 5.037
V_pppi = -3.033
#d
V_sdsi = -2.28
V_pdsi = -1.35
V_pdpi = 2.38

V_ddsi = -1.68
V_ddpi = 2.58
V_dddel = -1.81

epsilon = -8.868
epsilon_d = 10
"""

V_sssi = -6.769
#p
V_spsi = 5.58
V_ppsi = 5.037
V_pppi = -3.033
#d
V_sdsi = 0
V_pdsi = 0
V_pdpi = 1.5

V_ddsi = 0
V_ddpi = 0
V_dddel = 0

epsilon = -8.868
epsilon_d = 17

#co z resztą epsilonów?

######## PATH #####################
K1=[0,(4*np.pi)/(3*a*np.sqrt(3))]
K1_bis=[0,(4*np.pi)/(3*a*np.sqrt(3))-0.01]
Gam =[0,0]
M=[(np.pi)/(3*a),(np.pi)/(a*np.sqrt(3))]
K2=[0,(4*np.pi)/(3*a*np.sqrt(3))]
#ścieżka -K to daj wszędzie minusy
size = 100
path_1 = np.linspace(K1,Gam,size)
#path_1 = np.linspace(K1,K1_bis,size)
path_2 = np.linspace(Gam,M,size)
path_3 = np.linspace(M,K2,int(size/2))

path = np.append(path_1,path_2,axis=0)
path = np.append(path,path_3,axis=0)
#path = path_1

result_energies = []
result_eig_fun=[]

def H_A(K):

    H = np.zeros((9,9), dtype=complex)

    H[0,0] = V_sssi*(exp(1j*vdot(K,R1))+exp(1j*vdot(K,R2))+exp(1j*vdot(K,R3)))
    H[0,1] = V_spsi*(exp(1j*vdot(K,R1))-1/2*exp(1j*vdot(K,R2))-1/2*exp(1j*vdot(K,R3)))
    H[0,2] = V_spsi*(sqrt(3)/2)*(exp(1j*vdot(K,R2))-exp(1j*vdot(K,R3)))
    H[0,4] = V_sdsi*3/4*(-exp(1j*vdot(K,R2))+exp(1j*vdot(K,R3)))
    H[0,7] = V_sdsi*(sqrt(3)/2)*(exp(1j*vdot(K,R1))-1/2*exp(1j*vdot(K,R2))-1/2*exp(1j*vdot(K,R3)))
    H[0,8] = V_sdsi*(-1/2)*(exp(1j*vdot(K,R1))+exp(1j*vdot(K,R2))+exp(1j*vdot(K,R3)))

    H[1,0] = V_spsi*(-exp(1j*vdot(K,R1))+1/2*exp(1j*vdot(K,R2))+1/2*exp(1j*vdot(K,R3)))
    H[1,1] = V_ppsi*(exp(1j*vdot(K,R1))+1/4*exp(1j*vdot(K,R2))+1/4*exp(1j*vdot(K,R3)))+V_pppi*3/4*(exp(1j*vdot(K,R2))+exp(1j*vdot(K,R3)))
    H[1,2] = V_ppsi*sqrt(3)/4*(-exp(1j*vdot(K,R2))+exp(1j*vdot(K,R3)))+V_pppi*sqrt(3)/4*(exp(1j*vdot(K,R2))-exp(1j*vdot(K,R3)))
    H[1,4] = V_pdsi*3/8*(exp(1j*vdot(K,R2))-exp(1j*vdot(K,R3)))+V_pdpi*(sqrt(3)/4)*(exp(1j*vdot(K,R2))-exp(1j*vdot(K,R3)))
    H[1,7] = V_pdsi*(sqrt(3)/2)*(exp(1j*vdot(K,R1))+1/4*exp(1j*vdot(K,R2))+1/4*exp(1j*vdot(K,R3)))+V_pdpi*-3/4*(exp(1j*vdot(K,R2))+exp(1j*vdot(K,R3)))
    H[1,8] = V_pdsi*(-1/2*exp(1j*vdot(K,R1))+1/4*exp(1j*vdot(K,R2))+1/4*exp(1j*vdot(K,R3)))

    H[2,0] = V_spsi*(sqrt(3)/2)*(-exp(1j*vdot(K,R2))+exp(1j*vdot(K,R3)))
    H[2,1] = V_ppsi*(sqrt(3)/4)*(-exp(1j*vdot(K,R2))+exp(1j*vdot(K,R3)))+V_pppi*(sqrt(3)/4)*(exp(1j*vdot(K,R2))-exp(1j*vdot(K,R3)))
    H[2,2] = V_ppsi*(3/4)*(exp(1j*vdot(K,R2))+exp(1j*vdot(K,R3)))+V_pppi*(exp(1j*vdot(K,R1))+1/4*exp(1j*vdot(K,R2))+1/4*exp(1j*vdot(K,R3)))
    H[2,4] = V_pdsi*(-(3*sqrt(3))/8)*(exp(1j*vdot(K,R2))+exp(1j*vdot(K,R3)))+V_pdpi*(exp(1j*vdot(K,R1))+1/4*exp(1j*vdot(K,R2))+1/4*exp(1j*vdot(K,R3)))
    H[2,7] = V_pdsi*(3/8)*(-exp(1j*vdot(K,R2))+exp(1j*vdot(K,R3)))+V_pdpi*(sqrt(3)/4)*(-exp(1j*vdot(K,R2))+exp(1j*vdot(K,R3)))
    H[2,8] = V_pdsi*(sqrt(3)/4)*(-exp(1j*vdot(K,R2))+exp(1j*vdot(K,R3)))

    H[3,3] = V_pppi*(exp(1j*vdot(K,R1))+exp(1j*vdot(K,R2))+exp(1j*vdot(K,R3)))
    H[3,5] = V_pdpi*(sqrt(3)/2)*(exp(1j*vdot(K,R2))-exp(1j*vdot(K,R3)))
    H[3,6] = V_pdpi*(exp(1j*vdot(K,R1))-1/2*exp(1j*vdot(K,R2))-1/2*exp(1j*vdot(K,R3)))

    H[4,0] = V_sdsi*(3/4)*(-exp(1j*vdot(K,R2))+exp(1j*vdot(K,R3)))
    H[4,1] = V_pdsi*(3/8)*(-exp(1j*vdot(K,R2))+exp(1j*vdot(K,R3)))+V_pdpi*(sqrt(3)/4)*(-exp(1j*vdot(K,R2))+exp(1j*vdot(K,R3)))
    H[4,2] = V_pdsi*(3*(sqrt(3))/8)*(exp(1j*vdot(K,R2))+exp(1j*vdot(K,R3)))\
            +V_pdpi*(-exp(1j*vdot(K,R1))-1/4*exp(1j*vdot(K,R2))-1/4*exp(1j*vdot(K,R3)))
    H[4,4] = V_ddsi*(9/16)*(exp(1j*vdot(K,R2))+exp(1j*vdot(K,R3)))+V_ddpi*(exp(1j*vdot(K,R1))+1/4*exp(1j*vdot(K,R2))+1/4*exp(1j*vdot(K,R3)))\
            +V_dddel*3/16*(exp(1j*vdot(K,R2))+exp(1j*vdot(K,R3)))
    H[4,7] = V_ddsi*(3*sqrt(3))/16*(exp(1j*vdot(K,R2))-exp(1j*vdot(K,R3)))+V_ddpi*(-sqrt(3)/4)*exp(1j*vdot(K,R2))+sqrt(3)/4*exp(1j*vdot(K,R3))\
            +V_dddel*sqrt(3)/16*(exp(1j*vdot(K,R2))-exp(1j*vdot(K,R3)))
    H[4,8] = V_ddsi*(3/8)*(exp(1j*vdot(K,R2))-exp(1j*vdot(K,R3)))+V_dddel*(3/8)*(-exp(1j*vdot(K,R2))+exp(1j*vdot(K,R3)))

    H[5,3] = V_pdpi*(sqrt(3)/2)*(-exp(1j*vdot(K,R2))+exp(1j*vdot(K,R3)))
    H[5,5] = V_ddpi*(3/4)*(exp(1j*vdot(K,R2))+exp(1j*vdot(K,R3)))+V_dddel*(exp(1j*vdot(K,R1))+1/4*exp(1j*vdot(K,R2))+1/4*exp(1j*vdot(K,R3)))
    H[5,6] = V_ddpi*(sqrt(3)/4)*(-exp(1j*vdot(K,R2))+exp(1j*vdot(K,R3)))+V_dddel*(sqrt(3)/4)*(exp(1j*vdot(K,R2))-exp(1j*vdot(K,R3)))

    H[6,3] = V_pdpi*(-exp(1j*vdot(K,R1))+1/2*exp(1j*vdot(K,R2))+1/2*exp(1j*vdot(K,R3)))
    H[6,5] = V_ddpi*(sqrt(3)/4)*(-exp(1j*vdot(K,R2))+exp(1j*vdot(K,R3)))+V_dddel*(sqrt(3)/4)*(exp(1j*vdot(K,R2))-exp(1j*vdot(K,R3)))
    H[6,6] = V_ddpi*(exp(1j*vdot(K,R1))+1/4*exp(1j*vdot(K,R2))+1/4*exp(1j*vdot(K,R3)))+V_dddel*(3/4)*(exp(1j*vdot(K,R2))+exp(1j*vdot(K,R3)))

    H[7,0] = V_sdsi*(sqrt(3)/2)*(exp(1j*vdot(K,R1))-1/2*exp(1j*vdot(K,R2))-1/2*exp(1j*vdot(K,R3)))
    H[7,1] = V_pdsi*(-sqrt(3)/2)*(exp(1j*vdot(K,R1))+1/4*exp(1j*vdot(K,R2))+1/4*exp(1j*vdot(K,R3)))\
            +V_pdpi*3/4*(exp(1j*vdot(K,R2))+exp(1j*vdot(K,R3)))
    H[7,2] = V_pdsi*3/8*(exp(1j*vdot(K,R2))-exp(1j*vdot(K,R3)))+V_pdpi*(sqrt(3)/4)*(exp(1j*vdot(K,R2))-exp(1j*vdot(K,R3)))
    H[7,4] = V_ddsi*(3*sqrt(3)/16)*(exp(1j*vdot(K,R2))-exp(1j*vdot(K,R3)))+V_ddpi*(sqrt(3)/4)*(-exp(1j*vdot(K,R2))+exp(1j*vdot(K,R3)))\
            +V_dddel*(sqrt(3)/16)*(exp(1j*vdot(K,R2))-exp(1j*vdot(K,R3)))
    H[7,7] = V_ddsi*(3/4)*(exp(1j*vdot(K,R1))+1/4*exp(1j*vdot(K,R2))+1/4*exp(1j*vdot(K,R3))) \
             +V_ddpi*(-3/4)*(exp(1j*vdot(K,R2))+exp(1j*vdot(K,R3))) \
             +V_dddel*(1/4)*(exp(1j*vdot(K,R1))+1/4*exp(1j*vdot(K,R2))+1/4*exp(1j*vdot(K,R3)))
    H[7,8] = V_ddsi*(1/4)*(-exp(1j*vdot(K,R1))+1/2*exp(1j*vdot(K,R2))+1/2*exp(1j*vdot(K,R3))) \
             +V_dddel*(1/4)*(exp(1j*vdot(K,R1))-1/2*exp(1j*vdot(K,R2))-1/2*exp(1j*vdot(K,R3)))

    H[8,0] = V_sdsi*(-1/2)*(exp(1j*vdot(K,R1))+exp(1j*vdot(K,R2))+exp(1j*vdot(K,R3)))
    H[8,1] = V_pdsi*(1/2)*(exp(1j*vdot(K,R1))-1/2*exp(1j*vdot(K,R2))-1/2*exp(1j*vdot(K,R3)))
    H[8,2] = V_pdsi*(sqrt(3)/4)*(exp(1j*vdot(K,R2))-exp(1j*vdot(K,R3)))
    H[8,4] = V_ddsi*(3/8)*(exp(1j*vdot(K,R2))-exp(1j*vdot(K,R3)))+V_dddel*(3/8)*(-exp(1j*vdot(K,R2))+exp(1j*vdot(K,R3)))
    H[8,7] = V_ddsi*(-1/4*exp(1j*vdot(K,R1))+1/8*exp(1j*vdot(K,R2))+1/8*exp(1j*vdot(K,R3))) \
             +V_dddel*(1/4*exp(1j*vdot(K,R1))-1/8*exp(1j*vdot(K,R2))-1/8*exp(1j*vdot(K,R3)))
    H[8,8] =  V_ddsi*(1/4*exp(1j*vdot(K,R1))+1/4*exp(1j*vdot(K,R2))+1/4*exp(1j*vdot(K,R3))) \
             +V_dddel*(3/4*exp(1j*vdot(K,R1))+3/4*exp(1j*vdot(K,R2))+3/4*exp(1j*vdot(K,R3)))

    return H

def H_B(K):
    H = np.zeros((9,9), dtype=complex)

    H[0,0] = V_sssi*(exp(1j*vdot(K,R1m))+exp(1j*vdot(K,R2m))+exp(1j*vdot(K,R3m)))
    H[0,1] = V_spsi*(-exp(1j*vdot(K,R1m))+1/2*exp(1j*vdot(K,R2m))+1/2*exp(1j*vdot(K,R3m)))
    H[0,2] = V_spsi*(sqrt(3)/2)*(-exp(1j*vdot(K,R2m))+exp(1j*vdot(K,R3m)))
    H[0,4] = V_sdsi*(3/4)*(-exp(1j*vdot(K,R2m))+exp(1j*vdot(K,R3m)))
    H[0,7] = V_sdsi*(sqrt(3)/2)*(exp(1j*vdot(K,R1m))-1/2*exp(1j*vdot(K,R2m))-1/2*exp(1j*vdot(K,R3m)))
    H[0,8] = V_sdsi*(-1/2)*(exp(1j*vdot(K,R1m))+exp(1j*vdot(K,R2m))+exp(1j*vdot(K,R3m)))

    H[1,0] = V_spsi*(exp(1j*vdot(K,R1m))-1/2*exp(1j*vdot(K,R2m))-1/2*exp(1j*vdot(K,R3m)))
    H[1,1] = V_ppsi*(exp(1j*vdot(K,R1m))+1/4*exp(1j*vdot(K,R2m))+1/4*exp(1j*vdot(K,R3m)))+V_pppi*3/4*(exp(1j*vdot(K,R2m))+exp(1j*vdot(K,R3m)))
    H[1,2] = V_ppsi*(sqrt(3)/4)*(-exp(1j*vdot(K,R2m))+exp(1j*vdot(K,R3m)))+V_pppi*(sqrt(3)/4)*(exp(1j*vdot(K,R2m))-exp(1j*vdot(K,R3m)))
    H[1,4] = V_pdsi*(3/8)*(-exp(1j*vdot(K,R2m))+exp(1j*vdot(K,R3m)))+V_pdpi*(sqrt(3)/4)*(-exp(1j*vdot(K,R2m))+exp(1j*vdot(K,R3m)))
    H[1,7] = V_pdsi*(-sqrt(3)/2)*(exp(1j*vdot(K,R1m))+1/4*exp(1j*vdot(K,R2m))+1/4*exp(1j*vdot(K,R3m)))\
            +V_pdpi*3/4*(exp(1j*vdot(K,R2m))+exp(1j*vdot(K,R3m)))
    H[1,8] = V_pdsi*(1/2)*(exp(1j*vdot(K,R1m))-1/2*exp(1j*vdot(K,R2m))-1/2*exp(1j*vdot(K,R3m)))

    H[2,0] = V_spsi*(sqrt(3)/2)*(exp(1j*vdot(K,R2m))-exp(1j*vdot(K,R3m)))
    H[2,1] = V_ppsi*sqrt(3)/4*(-exp(1j*vdot(K,R2m))+exp(1j*vdot(K,R3m)))+V_pppi*sqrt(3)/4*(exp(1j*vdot(K,R2m))-exp(1j*vdot(K,R3m)))
    H[2,2] = V_ppsi*(3/4)*(exp(1j*vdot(K,R2m))+exp(1j*vdot(K,R3m)))+V_pppi*(exp(1j*vdot(K,R1m))+1/4*exp(1j*vdot(K,R2m))+1/4*exp(1j*vdot(K,R3m)))
    H[2,4] = V_pdsi*(3*(sqrt(3))/8)*(exp(1j*vdot(K,R2m))+exp(1j*vdot(K,R3m)))\
            +V_pdpi*(-exp(1j*vdot(K,R1m))-1/4*exp(1j*vdot(K,R2m))-1/4*exp(1j*vdot(K,R3m)))
    H[2,7] = V_pdsi*(3/8)*(exp(1j*vdot(K,R2m))-exp(1j*vdot(K,R3m)))+V_pdpi*(sqrt(3)/4)*(exp(1j*vdot(K,R2m))-exp(1j*vdot(K,R3m)))
    H[2,8] = V_pdsi*(sqrt(3)/4)*(exp(1j*vdot(K,R2m))-exp(1j*vdot(K,R3m)))

    H[3,3] = V_pppi*(exp(1j*vdot(K,R1m))+exp(1j*vdot(K,R2m))+exp(1j*vdot(K,R3m)))
    H[3,5] = V_pdpi*(sqrt(3)/2)*(-exp(1j*vdot(K,R2m))+exp(1j*vdot(K,R3m)))
    H[3,6] = V_pdpi*(-exp(1j*vdot(K,R1m))+1/2*exp(1j*vdot(K,R2m))+1/2*exp(1j*vdot(K,R3m)))

    H[4,0] = V_sdsi*3/4*(-exp(1j*vdot(K,R2m))+exp(1j*vdot(K,R3m)))
    H[4,1] = V_pdsi*3/8*(exp(1j*vdot(K,R2m))-exp(1j*vdot(K,R3m)))+V_pdpi*(sqrt(3)/4)*(exp(1j*vdot(K,R2m))-exp(1j*vdot(K,R3m)))
    H[4,2] = V_pdsi*(-(3*sqrt(3))/8)*(exp(1j*vdot(K,R2m))+exp(1j*vdot(K,R3m)))+V_pdpi*(exp(1j*vdot(K,R1m))+1/4*exp(1j*vdot(K,R2m))+1/4*exp(1j*vdot(K,R3m)))
    H[4,4] = V_ddsi*(9/16)*(exp(1j*vdot(K,R2m))+exp(1j*vdot(K,R3m)))+V_ddpi*(exp(1j*vdot(K,R1m))+1/4*exp(1j*vdot(K,R2m))+1/4*exp(1j*vdot(K,R3m)))\
            +V_dddel*3/16*(exp(1j*vdot(K,R2m))+exp(1j*vdot(K,R3m)))
    H[4,7] = V_ddsi*(3*sqrt(3))/16*(exp(1j*vdot(K,R2m))-exp(1j*vdot(K,R3m)))+V_ddpi*(-sqrt(3)/4)*exp(1j*vdot(K,R2m))+sqrt(3)/4*exp(1j*vdot(K,R3m))\
            +V_dddel*sqrt(3)/16*(exp(1j*vdot(K,R2m))-exp(1j*vdot(K,R3m)))
    H[4,8] = V_ddsi*(3/8)*(exp(1j*vdot(K,R2m))-exp(1j*vdot(K,R3m)))+V_dddel*(3/8)*(-exp(1j*vdot(K,R2m))+exp(1j*vdot(K,R3m)))

    H[5,3] = V_pdpi*(sqrt(3)/2)*(exp(1j*vdot(K,R2m))-exp(1j*vdot(K,R3m)))
    H[5,5] = V_ddpi*(3/4)*(exp(1j*vdot(K,R2m))+exp(1j*vdot(K,R3m)))+V_dddel*(exp(1j*vdot(K,R1m))+1/4*exp(1j*vdot(K,R2m))+1/4*exp(1j*vdot(K,R3m)))
    H[5,6] = V_ddpi*(sqrt(3)/4)*(-exp(1j*vdot(K,R2m))+exp(1j*vdot(K,R3m)))+V_dddel*(sqrt(3)/4)*(exp(1j*vdot(K,R2m))-exp(1j*vdot(K,R3m)))

    H[6,3] = V_pdpi*(exp(1j*vdot(K,R1m))-1/2*exp(1j*vdot(K,R2m))-1/2*exp(1j*vdot(K,R3m)))
    H[6,5] = V_ddpi*(sqrt(3)/4)*(-exp(1j*vdot(K,R2m))+exp(1j*vdot(K,R3m)))+V_dddel*(sqrt(3)/4)*(exp(1j*vdot(K,R2m))-exp(1j*vdot(K,R3m)))
    H[6,6] = V_ddpi*(exp(1j*vdot(K,R1m))+1/4*exp(1j*vdot(K,R2m))+1/4*exp(1j*vdot(K,R3m)))+V_dddel*(3/4)*(exp(1j*vdot(K,R2m))+exp(1j*vdot(K,R3m)))

    H[7,0] = V_sdsi*(sqrt(3)/2)*(exp(1j*vdot(K,R1m))-1/2*exp(1j*vdot(K,R2m))-1/2*exp(1j*vdot(K,R3m)))
    H[7,1] = V_pdsi*(sqrt(3)/2)*(exp(1j*vdot(K,R1m))+1/4*exp(1j*vdot(K,R2m))+1/4*exp(1j*vdot(K,R3m)))+V_pdpi*-3/4*(exp(1j*vdot(K,R2m))+exp(1j*vdot(K,R3m)))
    H[7,2] = V_pdsi*3/8*(-exp(1j*vdot(K,R2m))+exp(1j*vdot(K,R3m)))+V_pdpi*(sqrt(3)/4)*(-exp(1j*vdot(K,R2m))+exp(1j*vdot(K,R3m)))
    H[7,4] = V_ddsi*(3*sqrt(3)/16)*(exp(1j*vdot(K,R2m))-exp(1j*vdot(K,R3m)))+V_ddpi*(sqrt(3)/4)*(-exp(1j*vdot(K,R2m))+exp(1j*vdot(K,R3m)))\
            +V_dddel*(sqrt(3)/16)*(exp(1j*vdot(K,R2m))-exp(1j*vdot(K,R3m)))
    H[7,7] = V_ddsi*(3/4)*(exp(1j*vdot(K,R1m))+1/4*exp(1j*vdot(K,R2m))+1/4*exp(1j*vdot(K,R3m)))\
             +V_ddpi*(-3/4)*(exp(1j*vdot(K,R2m))+exp(1j*vdot(K,R3m)))\
             +V_dddel*(1/4)*(exp(1j*vdot(K,R1m))+1/4*exp(1j*vdot(K,R2m))+1/4*exp(1j*vdot(K,R3m)))
    H[7,8] = V_ddsi*(1/4)*(-exp(1j*vdot(K,R1m))+1/2*exp(1j*vdot(K,R2m))+1/2*exp(1j*vdot(K,R3m)))\
            +V_dddel*(1/4)*(exp(1j*vdot(K,R1m))-1/2*exp(1j*vdot(K,R2m))-1/2*exp(1j*vdot(K,R3m)))

    H[8,0] = V_sdsi*(-1/2)*(exp(1j*vdot(K,R1m))+exp(1j*vdot(K,R2m))+exp(1j*vdot(K,R3m)))
    H[8,1] = V_pdsi*(-1/2*exp(1j*vdot(K,R1m))+1/4*exp(1j*vdot(K,R2m))+1/4*exp(1j*vdot(K,R3m)))
    H[8,2] = V_pdsi*(sqrt(3)/4)*(-exp(1j*vdot(K,R2m))+exp(1j*vdot(K,R3m)))

    H[8,4] = V_ddsi*(3/8)*(exp(1j*vdot(K,R2m))-exp(1j*vdot(K,R3m)))+V_dddel*(3/8)*(-exp(1j*vdot(K,R2m))+exp(1j*vdot(K,R3m)))
    H[8,7] = V_ddsi*(-1/4*exp(1j*vdot(K,R1m))+1/8*exp(1j*vdot(K,R2m))+1/8*exp(1j*vdot(K,R3m)))\
             +V_dddel*(1/4*exp(1j*vdot(K,R1m))-1/8*exp(1j*vdot(K,R2m))-1/8*exp(1j*vdot(K,R3m)))
    H[8,8] =  V_ddsi*(1/4*exp(1j*vdot(K,R1m))+1/4*exp(1j*vdot(K,R2m))+1/4*exp(1j*vdot(K,R3m)))\
             +V_dddel*(3/4*exp(1j*vdot(K,R1m))+3/4*exp(1j*vdot(K,R2m))+3/4*exp(1j*vdot(K,R3m)))

    return H

def full_H(k):
    H11 = np.zeros((9,9), dtype=complex)
    H22 = np.zeros((9,9), dtype=complex)
    H11[0][0] = epsilon
    H11[4][4] = epsilon_d
    H11[5][5] = epsilon_d
    H11[6][6] = epsilon_d
    H11[7][7] = epsilon_d
    H11[8][8] = epsilon_d


    H22[0][0] = epsilon
    H22[4][4] = epsilon_d
    H22[5][5] = epsilon_d
    H22[6][6] = epsilon_d
    H22[7][7] = epsilon_d
    H22[8][8] = epsilon_d

    H12 = H_A(k)
    H21 = H_B(k)
    H = np.block([[H11, H12], [H21, H22]])
    return H

def check_symmetric(a):
    if not np.allclose(a, np.asmatrix(a).H):
        print("error")
        raise ValueError('expected symmetric or Hermitian matrix,\
        try using numpy.linalg.eig instead')

    else: print("ok")

def norm_2(z):
    #modulus of complex vector
    com_num = (z.real)**2 + (z.imag)**2
    return com_num

def check_complex(com_number,value):
    arr =[]
    for i in range(len(com_number)):
        check = com_number[i].imag
        if check >= value:
            arr.append(check)
    return arr

#creating files to save data
'''
f = open("eig_vec_complex.txt",'w')
f.write("K_point \t Energy \t Eigenvector\n")
f_2 = open("eig_vec_real.txt",'w')
f_2.write("K_point \t Energy \t Eigenvector\n")
'''
f_3 = open("sp3d5_real_results_epsilon"+str(epsilon_d)+".txt",'w')
#f_3 = open("sp3d5_results_bis"+".txt",'w')
#f_3.write("K_point \t Energy \t Eigenvector_modulus \t Sum\n")

for i in range(len(path)):
    #exact diagonalization

    #check_symmetric(full_H(path[i]))
    energies, vectors = np.linalg.eig(full_H(path[i]))

    #check value of imag part
    arr = check_complex(energies,10e-10)
    if arr != []:
        print(arr)

    #list for plotting
    result_energies.append(energies)
    #print(energies)
    #print(vectors)

    for j in range(len(energies)):
        #specific writing to files
        eig_modul = []
        #f.write(str(i)+' ')
        #f_2.write(str(i)+' ')
        f_3.write(str(i)+' ')
        #data = str(energies[j])
        data_ = str(energies[j].real)
        #f.write((' '+data+' '))
        #f_2.write(' '+data_+' ')
        f_3.write(' '+data_+' ')
        for k in range(len(vectors[j])):
            modul = norm_2(vectors[k][j])
            #data_1 =[vectors[k][j],' ']
            #data_2 = [vectors[k][j].real,' ']
            data_3 = [modul,' ']
            eig_modul.append(modul)
            #f.write((' '.join(map(str,data_1))))
            #f_2.write((' '.join(map(str,data_2))))
            f_3.write((' '.join(map(str,data_3))))
        #f.write('\n')
        #f_2.write('\n')
        suma = sum(eig_modul)
        f_3.write(str(suma))
        f_3.write('\n')
        #print(suma)
#f.close()
#f_2.close()
f_3.close()

#if you use .eig you need to sort energies by complex before plotting
result_energies = np.sort_complex(result_energies)

### plotting band structure#####
x = np.linspace(0,1,len(path))
c_1,c_2,c_3,c_4,c_5,c_6,c_7,c_8,c_9,c_10,c_11,c_12,c_13,c_14,c_15,c_16,c_17,c_18\
         = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]

for i in range(len(result_energies)):
    c_1.append(result_energies[i][0])
    c_2.append(result_energies[i][1])
    c_3.append(result_energies[i][2])
    c_4.append(result_energies[i][3])
    c_5.append(result_energies[i][4])
    c_6.append(result_energies[i][5])
    c_7.append(result_energies[i][6])
    c_8.append(result_energies[i][7])
    c_9.append(result_energies[i][8])
    c_10.append(result_energies[i][9])
    c_11.append(result_energies[i][10])
    c_12.append(result_energies[i][11])
    c_13.append(result_energies[i][12])
    c_14.append(result_energies[i][13])
    c_15.append(result_energies[i][14])
    c_16.append(result_energies[i][15])
    c_17.append(result_energies[i][16])
    c_18.append(result_energies[i][17])


fig, ax = plt.subplots()
fig.suptitle("sp3d5 real basis")
ax.plot(x,c_1)
ax.plot(x,c_2)
ax.plot(x,c_3)
ax.plot(x,c_4)
ax.plot(x,c_5)
ax.plot(x,c_6)
ax.plot(x,c_7)
ax.plot(x,c_8)
ax.plot(x,c_9)
ax.plot(x,c_10)
ax.plot(x,c_11)
ax.plot(x,c_12)
ax.plot(x,c_13)
ax.plot(x,c_14)
ax.plot(x,c_15)
ax.plot(x,c_16)
ax.plot(x,c_17)
ax.plot(x,c_18)

ax.set_xticks(np.linspace(0,1,6))
ax.set_xticklabels(["K","","$\Gamma$","","M","K"])
plt.show()
print(energies)
print("sp3d5 ended")
