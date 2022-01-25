###### real basis #########
import matplotlib.pyplot as plt
import numpy as np
from numpy import sqrt

#data:
a = 1.42
R1B = np.array([a,0])
R2B = np.array([-a/2,(a*np.sqrt(3))/2])
R3B = np.array([-a/2,-(a*np.sqrt(3))/2])

R1A =np.array([-a,0])
R2A =np.array([a/2,-(a*np.sqrt(3))/2])
R3A =np.array([a/2,(a*np.sqrt(3))/2])

R1 = [a, 0]
R1m = [-a, 0]
R2 = [-a/2, sqrt(3)*a/2]
R2m = [a/2, -a*sqrt(3)/2]
R3 = [-a/2, -sqrt(3)*a/2]
#tu jest difference
R3m = [a/2, a*sqrt(3)/2]

V_sssi = -6.769
V_spsi = 5.58
V_ppsi = 5.037
V_pppi = -3.033
epsilon = -8.868

######## PATH ############
x1 = 0
x2 = 0
x3 = np.pi/(3*a)
x4 = 0

y1 =(4*np.pi)/(3*a*np.sqrt(3))
y2 = 0
y3 = np.pi/(a*np.sqrt(3))
y4 = (4*np.pi)/(3*a*np.sqrt(3))

size = 100
# K - Gam - M - K
path_1 = np.linspace((x1,y1),(x2,y2),size)
path_2 = np.linspace((x2,y2),(x3,y3),size)
path_3 = np.linspace((x3,y3),(x4,y4),int(size/2))

#Gam - K - M - Gam
#path_1 = np.linspace((x2,y2),(x1,y1),size)
#path_2 = np.linspace((x1,y1),(x3,y3),int(size/2))
#path_3 = np.linspace((x3,y3),(x2,y2),size)

#path = np.append(path_1,path_2,axis=0)
#path = np.append(path,path_3,axis=0)

#only around K 
#path = [x1,y1]


####### HAMILTONIAN ##############


result = []

def H_A(K):
    H = np.zeros((4,4), dtype=complex)
    H[0,0]= V_sssi*(np.exp(1j*np.vdot(K,R1))+np.exp(1j*np.vdot(K,R2))+np.exp(1j*np.vdot(K,R3)))
    H[1,0]= V_spsi*(-np.exp(1j*np.vdot(K,R1))+1/2*np.exp(1j*np.vdot(K,R2))+1/2*np.exp(1j*np.vdot(K,R3)))
    H[2,0]= V_spsi*(np.sqrt(3)/2)*(-np.exp(1j*np.vdot(K,R2))+np.exp(1j*np.vdot(K,R3)))

    H[0,1]= V_spsi*(np.exp(1j*np.vdot(K,R1))-1/2*np.exp(1j*np.vdot(K,R2))-1/2*np.exp(1j*np.vdot(K,R3)))
    H[1,1]= V_ppsi*(np.exp(1j*np.vdot(K,R1))+1/4*np.exp(1j*np.vdot(K,R2))+1/4*np.exp(1j*np.vdot(K,R3)))\
    +V_pppi*3/4*(np.exp(1j*np.vdot(K,R2))+np.exp(1j*np.vdot(K,R3)))
    H[2,1]= V_ppsi*np.sqrt(3)/4*(-np.exp(1j*np.vdot(K,R2))+np.exp(1j*np.vdot(K,R3)))+V_pppi*(np.sqrt(3)/4)*(np.exp(1j*np.vdot(K,R2))-np.exp(1j*np.vdot(K,R3)))

    H[0,2]= V_spsi*(np.sqrt(3)/2)*(np.exp(1j*np.vdot(K,R2B))-np.exp(1j*np.vdot(K,R3B)))
    H[1,2]= V_ppsi*np.sqrt(3)/4*(-np.exp(1j*np.vdot(K,R2))+np.exp(1j*np.vdot(K,R3)))+V_pppi*(np.sqrt(3)/4)*(np.exp(1j*np.vdot(K,R2))-np.exp(1j*np.vdot(K,R3)))
    H[2,2]= V_pppi*(np.exp(1j*np.vdot(K,R1))+1/4*np.exp(1j*np.vdot(K,R2))+1/4*np.exp(1j*np.vdot(K,R3)))\
    +V_ppsi*3/4*(np.exp(1j*np.vdot(K,R2))+np.exp(1j*np.vdot(K,R3)))

    H[3,3]= V_pppi*(np.exp(1j*np.vdot(K,R1))+np.exp(1j*np.vdot(K,R2))+np.exp(1j*np.vdot(K,R3)))
    return H


def H_B(K):
    H = np.zeros((4,4), dtype=complex)
    H[0,0]= V_sssi*(np.exp(1j*np.vdot(K,R1m))+np.exp(1j*np.vdot(K,R2m))+np.exp(1j*np.vdot(K,R3m)))
    H[1,0]= V_spsi*(np.exp(1j*np.vdot(K,R1m))-1/2*np.exp(1j*np.vdot(K,R2m))-1/2*np.exp(1j*np.vdot(K,R3m)))
    H[2,0]= V_spsi*(np.sqrt(3)/2)*(np.exp(1j*np.vdot(K,R2m))-np.exp(1j*np.vdot(K,R3m)))

    H[0,1]= V_spsi*(-np.exp(1j*np.vdot(K,R1m))+1/2*np.exp(1j*np.vdot(K,R2m))+1/2*np.exp(1j*np.vdot(K,R3m)))
    H[1,1]= V_ppsi*(np.exp(1j*np.vdot(K,R1m))+1/4*np.exp(1j*np.vdot(K,R2m))+1/4*np.exp(1j*np.vdot(K,R3m)))\
    +V_pppi*3/4*(np.exp(1j*np.vdot(K,R2m))+np.exp(1j*np.vdot(K,R3m)))
    H[2,1]= V_ppsi*(np.sqrt(3)/4)*(-np.exp(1j*np.vdot(K,R2m))+np.exp(1j*np.vdot(K,R3m)))+\
    V_pppi*(np.sqrt(3)/4)*(np.exp(1j*np.vdot(K,R2m))-np.exp(1j*np.vdot(K,R3m)))

    H[0,2]= V_spsi*(np.sqrt(3)/2)*(-np.exp(1j*np.vdot(K,R2m))+np.exp(1j*np.vdot(K,R3m)))
    H[1,2]= V_ppsi*(np.sqrt(3)/4)*(-np.exp(1j*np.vdot(K,R2m))+np.exp(1j*np.vdot(K,R3m)))+\
    V_pppi*(np.sqrt(3)/4)*(np.exp(1j*np.vdot(K,R2m))-np.exp(1j*np.vdot(K,R3m)))
    H[2,2]= V_pppi*(np.exp(1j*np.vdot(K,R1m))+1/4*np.exp(1j*np.vdot(K,R2m))+1/4*np.exp(1j*np.vdot(K,R3m)))\
    +V_ppsi*3/4*(np.exp(1j*np.vdot(K,R2m))+np.exp(1j*np.vdot(K,R3m)))


    H[3,3]= V_pppi*(np.exp(1j*np.vdot(K,R1m))+np.exp(1j*np.vdot(K,R2m))+np.exp(1j*np.vdot(K,R3m)))

    return H

def full_H(k,lam):
    H11 = np.zeros((4,4), dtype=complex)
    H22 = np.zeros((4,4), dtype=complex)
    H11[0][0] = epsilon
    H11[1][2] = -1j/2*lam
    H11[2][1] = 1j/2*lam

    H22[0][0] = epsilon
    H22[1][2] = 1j/2*lam
    H22[2][1] = -1j/2*lam
    #here is funny
    #for i in range(0,1):
        #H11[i][i] = epsilon
        #H22[i][i] = epsilon
    H12 = H_A(k)
    H21 = H_B(k)
    H = np.block([[H11, H12], [H21, H22]])
    return H

def H_spin_up_down(lam): 
    H_zeros = np.zeros((4,4), dtype=complex)
    H_spin = np.zeros((4,4), dtype=complex)
    H_spin[1,3] = 1/2*lam
    H_spin[2,3] = -1j/2*lam
    H_spin[3,1] = -1/2*lam
    H_spin[3,2] = 1j/2*lam

    H = np.block([[H_spin, H_zeros], [H_zeros, H_spin]])

    return H

def H_spin_dwon_up(lam): 
    H_zeros = np.zeros((4,4), dtype=complex)
    H_spin = np.zeros((4,4), dtype=complex)
    H_spin[1,3] = -1/2*lam
    H_spin[2,3] = -1j/2*lam
    H_spin[3,1] = 1/2*lam
    H_spin[3,2] = 1j/2*lam

    H = np.block([[H_spin, H_zeros], [H_zeros, H_spin]])

    return H

def full_H_spin(k,lam):
    Hup = full_H(k,lam)
    Hdown = full_H(k,lam)
    #H_zeros = np.zeros((8,8), dtype=complex)
    H_spin_1 = H_spin_dwon_up(lam)
    H_spin_2 = H_spin_up_down(lam)
    
    #H = np.block([[Hup, H_zeros], [H_zeros, Hdown]])
    H = np.block([[Hup, H_spin_1], [H_spin_2, Hdown]])
    return(H)

def check_symmetric(a):
    if not np.allclose(a, np.asmatrix(a).H):
        print("error")
        raise ValueError('expected symmetric or Hermitian matrix,\
        try using numpy.linalg.eig instead')

    else: print("ok")

path = np.linspace((-1,y1),(0,y1),200)
path = np.append(path,np.linspace((0.00504,y1),(1,y1),200),axis=0)
lam = 1.5

f_3 = open("sp3_soc_up"+str(lam)+".txt",'w')
f_3.write("K_point Energy\n")
f_4 = open("sp3_soc_down"+str(lam)+".txt",'w')
f_4.write("K_point Energy\n")
for i in range(len(path)):
    #check_symmetric(full_H(path[i]))
    energies, vectors = np.linalg.eigh(full_H_spin(path[i],lam))
    result.append(energies)
    #print(energies)

    #specific band: 

    f_3.write(str(i)+' ')
    data_1 = str(energies[7].real)
    f_3.write(' '+data_1+' ')
    f_3.write('\n')

    f_4.write(str(i)+' ')
    data_2 = str(energies[8].real)
    f_4.write(' '+data_2+' ')
    f_4.write('\n')
    '''
    for j in range(len(energies)):
        #specific writing to files
        #eig_modul = []
        #f.write(str(i)+' ')
        #f_2.write(str(i)+' ')
        f_3.write(str(i)+' ')
        #data = str(energies[j])
        data_ = str(energies[j].real)
        #f.write((' '+data+' '))
        #f_2.write(' '+data_+' ')
        f_3.write(' '+data_+' ')
        f_3.write('\n')

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
        '''
        
#f.close()
#f_2.close()
f_3.close()
f_4.close()
#result=np.sort_complex(result)
### plotting #####
x = np.linspace(0,1,len(path))
c_1,c_2,c_3,c_4,c_5,c_6,c_7,c_8,c_9,c_10,c_11,c_12,c_13,c_14,c_15,c_16 = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]


for i in range(len(result)):
    c_1.append(result[i][0])
    c_2.append(result[i][1])
    c_3.append(result[i][2])
    c_4.append(result[i][3])
    c_5.append(result[i][4])
    c_6.append(result[i][5])
    c_7.append(result[i][6])
    c_8.append(result[i][7])
    c_9.append(result[i][8])
    c_10.append(result[i][9])
    c_11.append(result[i][10])
    c_12.append(result[i][11])
    c_13.append(result[i][12])
    c_14.append(result[i][13])
    c_15.append(result[i][14])
    c_16.append(result[i][15])

#plt.style.use(['science','notebook'])
fig, ax = plt.subplots()
#fig.suptitle("sp3 real basis")
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
ax.set_xticks(np.linspace(0,1,6))
ax.set_xticklabels(["K","","$\Gamma$","","M","K"])
plt.show()
