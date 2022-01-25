###### continnum model TBG Bernevig#########
import matplotlib.pyplot as plt
import numpy as np
from numpy import sqrt, dot, exp, conj, pi, sin, vdot


def even_parts(v, vv, parts):
    return zip(np.linspace(v[0], vv[0], parts+1), np.linspace(v[1], vv[1], parts+1))

def check_symmetric(a):
    if not np.allclose(a, np.asmatrix(a).H):
        print("error")
        raise ValueError('expected symmetric or Hermitian matrix,\
        try using numpy.linalg.eig instead')

    else: print("ok")

def norm_1(z):
    #modulus of complex vector
    com_num = (abs(z)**2).real
    return com_num

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

a0 = 2.46 #A
theta = 1.8/180*pi
#print(theta)
k_theta = 8*pi/(3*a0)*sin(theta/2)
hv = -6.11 #eV*nm
w = 0.110 #eV
u0 = 1

#path
K_m = [0,0]
#K_m_bis = [sqrt(3)/2*k_theta,-1/2*k_theta]
Gam_m = [-sqrt(3)/2*k_theta,-1/2*k_theta]

#może +M_M?
M_m = [0,-k_theta/2]
#M_m_bis = [-sqrt(3)/4*k_theta,k_theta/4]

#LINLIU
K_bis = [0,-k_theta]
K_bis_bis = [-sqrt(3)/2*k_theta,-3/2*k_theta]

####################################
size = 100
#Bernevig
path = list(even_parts(Gam_m,K_m,100))+list(even_parts(K_m,M_m,50))+list(even_parts(M_m,Gam_m,100))
path_name = ["$\Gamma$","","K","M","","$\Gamma$"]

#LinLIU
#path = list(even_parts(Gam_m,K_bis,50))+list(even_parts(K_bis,K_bis_bis,50))\
#+list(even_parts(K_bis_bis,K_m,100))+list(even_parts(K_m,Gam_m,50))
#path_name = ["$\Gamma$","K1","K2","","K0","$\Gamma$"]
####################################

#Q vectors
q1 = [0,-k_theta]
q2 = [sqrt(3)/2*k_theta,1/2*k_theta]
q3 = [-sqrt(3)/2*k_theta,1/2*k_theta]


def full_H_spin(K,lambda_L,lambda_R):

    #hamiltonian z conj
    H = np.zeros((16,16), dtype=complex)

    [kx,ky]=K

    #hK(k-q) matrices
    h_theta_12 = hv*exp(-1j*theta/2)*(kx+1j*ky)
    h_theta_21 = hv*exp(1j*theta/2)*(kx-1j*ky)
    h_theta_12_q1 = hv*exp(1j*theta/2)*((kx-q1[0])+1j*(ky-q1[1]))
    h_theta_21_q1 = hv*exp(-1j*theta/2)*((kx-q1[0])-1j*(ky-q1[1]))
    h_theta_12_q2 = hv*exp(1j*theta/2)*((kx-q2[0])+1j*(ky-q2[1]))
    h_theta_21_q2 = hv*exp(-1j*theta/2)*((kx-q2[0])-1j*(ky-q2[1]))
    h_theta_12_q3 = hv*exp(1j*theta/2)*((kx-q3[0])+1j*(ky-q3[1]))
    h_theta_21_q3 = hv*exp(-1j*theta/2)*((kx-q3[0])-1j*(ky-q3[1]))

    #H_top with spin
    H[0,0] = 1/2*lambda_L
    H[0,1] = h_theta_12
    H[1,0] = h_theta_21
    H[1,1] = 1/2*lambda_L
    H[2,2] = -1/2*lambda_L
    H[2,3] = h_theta_12 
    H[3,2] = h_theta_21
    H[3,3] = -1/2*lambda_L

    H[1,2] = -1j*lambda_R
    H[2,1] = 1j*lambda_R


    #H_bottom
    H[4,5] = h_theta_12_q1
    H[5,4] = h_theta_21_q1
    H[6,7] = h_theta_12_q1
    H[7,6] = h_theta_21_q1

    H[8,9] = h_theta_12_q2
    H[9,8] = h_theta_21_q2
    H[10,11] = h_theta_12_q2
    H[11,10] = h_theta_21_q2

    H[12,13] = h_theta_12_q3
    H[13,12] = h_theta_21_q3
    H[14,15] = h_theta_12_q3
    H[15,14] = h_theta_21_q3


    #T1 matrices
    H[0,4]=w*u0
    H[0,5]=w
    H[1,4]=w
    H[1,5]=w*u0
    H[4,0]=conj(H[0,4])
    H[5,0]=conj(H[0,5])
    H[4,1]=conj(H[1,4])
    H[5,1]=conj(H[1,5])

    H[2,6]=w*u0
    H[2,7]=w
    H[3,6]=w
    H[3,7]=w*u0
    H[6,2]=conj(H[2,6])
    H[7,2]=conj(H[2,7])
    H[6,3]=conj(H[3,6])
    H[7,3]=conj(H[3,7])

    #T2 matrices
    H[0,8]=w*u0
    H[0,9]=w*(-1/2+1j*sqrt(3)/2)
    H[1,8]=w*(-1/2-1j*sqrt(3)/2)
    H[1,9]=w*u0
    H[8,0]=conj(H[0,8])
    H[9,0]=conj(H[0,9])
    H[8,1]=conj(H[1,8])
    H[9,1]=conj(H[1,9])

    H[2,10]=w*u0
    H[2,11]=w*(-1/2+1j*sqrt(3)/2)
    H[3,10]=w*(-1/2-1j*sqrt(3)/2)
    H[3,11]=w*u0
    H[10,2]=conj(H[2,10])
    H[11,2]=conj(H[2,11])
    H[10,3]=conj(H[3,10])
    H[11,3]=conj(H[3,11])

    #T3 matrices
    H[0,12]=w*u0
    H[0,13]=w*(-1/2-1j*sqrt(3)/2)
    H[1,12]=w*(-1/2+1j*sqrt(3)/2)
    H[1,13]=w*u0
    H[12,0]=conj(H[0,12])
    H[13,0]=conj(H[0,13])
    H[12,1]=conj(H[1,12])
    H[13,1]=conj(H[1,13])

    H[2,14]=w*u0
    H[2,15]=w*(-1/2-1j*sqrt(3)/2)
    H[3,14]=w*(-1/2+1j*sqrt(3)/2)
    H[3,15]=w*u0
    H[14,2]=conj(H[2,14])
    H[15,2]=conj(H[2,15])
    H[14,3]=conj(H[3,14])
    H[15,3]=conj(H[3,15])

    return H



def h_with_saving(path,lambda_L,lambda_R):
    #creating files to save data
    f_3 = open("TBG_spin_"+str('%1.1f'%(theta*180/pi))+"_SOC_"+str(lambda_L)+"_R_"+str(lambda_R)+".txt",'w')
    f_3.write("K_point \t Energy \t Eigenvector_modulus \t Sum\n")

    for i in range(len(path)):
        #exact diagonalization

        #check_symmetric(full_H(path[i]))
        energies, vectors = np.linalg.eig(full_H_spin(path[i],lambda_L,lambda_R))
        

        for j in range(len(energies)):
            #specific writing to files
            eig_modul = []
            f_3.write(str(i)+' ')
            data_ = str(energies[j].real)
            f_3.write(' '+data_+' ')

            for k in range(len(vectors[j])):
                modul = norm_2(vectors[k][j])
                data_3 = [modul,' ']
                eig_modul.append(modul)
                f_3.write((' '.join(map(str,data_3))))
            suma = sum(eig_modul)
            f_3.write(str(suma))
            f_3.write('\n')
    f_3.close()
    print("TBG_ended")


#this not probably
def h_with_saving_spin_clac_divide(path,lambda_L,lambda_R):
    #creating files to save data
    f_3 = open("TBG_spin_up_"+str('%1.1f'%(theta*180/pi))+"_SOC_"+str(lambda_L)+"_R_"+str(lambda_R)+".txt",'w')
    f_4 = open("TBG_spin_down_"+str('%1.1f'%(theta*180/pi))+"_SOC_"+str(lambda_L)+"_R_"+str(lambda_R)+".txt",'w')
    f_3.write("K_point Energy spin_up \n")
    f_4.write("K_point Energy spin_dwon \n")
    
    for i in range(len(path)):
        #exact diagonalization
        #check_symmetric(full_H(path[i]))
        energies, vectors = np.linalg.eigh(full_H_spin(path[i],lambda_L,lambda_R))
        
        for j in range(len(energies)):
            #specific writing to files
            f_3.write(str(i)+' ')
            data_ = str(energies[j].real)
            f_3.write(' '+data_+' ')
            f_4.write(str(i)+' ')
            data_ = str(energies[j].real)
            f_4.write(' '+data_+' ')
            
            spin_up = dot(vectors[0,j],conj(vectors[0,j]))+dot(vectors[1,j],conj(vectors[1,j]))+dot(vectors[2,j],conj(vectors[2,j]))+dot(vectors[3,j],conj(vectors[3,j]))\
                +dot(vectors[4,j],conj(vectors[4,j]))+dot(vectors[5,j],conj(vectors[5,j]))+dot(vectors[6,j],conj(vectors[6,j]))+dot(vectors[7,j],conj(vectors[7,j]))
            spin_down = dot(vectors[8,j],conj(vectors[8,j]))+dot(vectors[9,j],conj(vectors[9,j]))+dot(vectors[10,j],conj(vectors[10,j]))+dot(vectors[11,j],conj(vectors[11,j]))\
                +dot(vectors[12,j],conj(vectors[12,j]))+dot(vectors[13,j],conj(vectors[13,j]))+dot(vectors[14,j],conj(vectors[14,j]))+dot(vectors[15,j],conj(vectors[15,j]))
            sum = (spin_up + spin_down) 

            vec_1 = norm_2(vectors[0][j])
            vec_2 = norm_2(vectors[1][j])
            vec_3 = norm_2(vectors[2][j])
            vec_4 = norm_2(vectors[3][j])
            vec_5 = norm_2(vectors[4][j])
            vec_6 = norm_2(vectors[5][j])
            vec_7 = norm_2(vectors[6][j])
            vec_8 = norm_2(vectors[7][j])
            vec_9 = norm_2(vectors[8][j])
            vec_10 = norm_2(vectors[9][j])
            vec_11 = norm_2(vectors[10][j])
            vec_12 = norm_2(vectors[11][j])
            vec_13 = norm_2(vectors[12][j])
            vec_14 = norm_2(vectors[13][j])
            vec_15 = norm_2(vectors[14][j])
            vec_16 = norm_2(vectors[15][j])

            sum_up = vec_1 + vec_2 + vec_5 + vec_6 + vec_9 + vec_10 + vec_13 + vec_14 
            sum_down = vec_3 + vec_4 + vec_7 + vec_8 + vec_11 + vec_12 + vec_15 + vec_16 

            f_3.write(' '+str(sum_up))
            f_4.write(' '+str(sum_down))
            #f_3.write(' '+str(sum))
            f_3.write('\n')
            f_4.write('\n')
    f_3.close()
    f_4.close()
    print("TBG_ended")


def h_with_spin_calculation(path,lambda_L,lambda_R):
    #creating files to save data
    f_3 = open("TBG_spin_"+str('%1.1f'%(theta*180/pi))+"_SOC_"+str(lambda_L)+"_R_"+str(lambda_R)+".txt",'w')
    f_3.write("K_point Energy eigenvectors spin_up spin_down Sum\n")

    for i in range(len(path)):
        #exact diagonalization
        #check_symmetric(full_H(path[i]))
        energies, vectors = np.linalg.eigh(full_H_spin(path[i],lambda_L,lambda_R))
        
        for j in range(len(energies)):
            #specific writing to files
            eig_modul = []
            f_3.write(str(i)+' ')
            data_ = str(energies[j].real)
            f_3.write(' '+data_+' ')
            
            spin_up = dot(vectors[0,j],conj(vectors[0,j]))+dot(vectors[1,j],conj(vectors[1,j]))+dot(vectors[2,j],conj(vectors[2,j]))+dot(vectors[3,j],conj(vectors[3,j]))\
                +dot(vectors[4,j],conj(vectors[4,j]))+dot(vectors[5,j],conj(vectors[5,j]))+dot(vectors[6,j],conj(vectors[6,j]))+dot(vectors[7,j],conj(vectors[7,j]))
            spin_down = dot(vectors[8,j],conj(vectors[8,j]))+dot(vectors[9,j],conj(vectors[9,j]))+dot(vectors[10,j],conj(vectors[10,j]))+dot(vectors[11,j],conj(vectors[11,j]))\
                +dot(vectors[12,j],conj(vectors[12,j]))+dot(vectors[13,j],conj(vectors[13,j]))+dot(vectors[14,j],conj(vectors[14,j]))+dot(vectors[15,j],conj(vectors[15,j]))
            sum = (spin_up + spin_down) 

            vec_1 = norm_2(vectors[0][j])
            vec_2 = norm_2(vectors[1][j])
            vec_3 = norm_2(vectors[2][j])
            vec_4 = norm_2(vectors[3][j])
            vec_5 = norm_2(vectors[4][j])
            vec_6 = norm_2(vectors[5][j])
            vec_7 = norm_2(vectors[6][j])
            vec_8 = norm_2(vectors[7][j])
            vec_9 = norm_2(vectors[8][j])
            vec_10 = norm_2(vectors[9][j])
            vec_11 = norm_2(vectors[10][j])
            vec_12 = norm_2(vectors[11][j])
            vec_13 = norm_2(vectors[12][j])
            vec_14 = norm_2(vectors[13][j])
            vec_15 = norm_2(vectors[14][j])
            vec_16 = norm_2(vectors[15][j])

            sum_up = vec_1 + vec_2 + vec_5 + vec_6 + vec_9 + vec_10 + vec_13 + vec_14 
            sum_down = vec_3 + vec_4 + vec_7 + vec_8 + vec_11 + vec_12 + vec_15 + vec_16 

            f_3.write(' '+str(sum_up))
            f_3.write(' '+str(sum_down))
            #f_3.write(' '+str(sum))
            f_3.write('\n')
    
    f_3.close()
    #print("Energies ",energies[1])
    #print("Vecrors: ",vectors[:,1])
    print("TBG_ended")


#lambda_L = 0.3
#lambda_R = 0.2
#h_with_spin_calculation(path,lambda_L,lambda_R)


# to do printu w pythonie

result_energies = []
result_eig_fun=[]
lambda_L=0.0
lambda_R=0.1

for i in range(len(path)):
    #exact diagonalization
    #check_symmetric(full_H(path[i]))
    #energies, vectors = np.linalg.eig(full_H(path[i]))
    energies, vectors = np.linalg.eig(full_H_spin(path[i],lambda_L,lambda_R))
    #checking value of imag part
    #arr = check_complex(energies,10e-10)
    #if arr != []:
        #print(arr)
    #list for plotting
    result_energies.append(energies)
    #print(energies)
    #print(vectors)


#if you use .eig you need to sort energies by complex before plotting
result_energies = np.sort_complex(result_energies)

### plotting band structure#####
x = np.linspace(0,1,len(path))
c_1,c_2,c_3,c_4,c_5,c_6,c_7,\
c_8,c_9,c_10,c_11,c_12,c_13,c_14\
 = [],[],[],[],[],[],[],[],[],[],[],[],[],[]

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

#for checking
#plt.imshow(full_H(path[100]).real)

fig, ax = plt.subplots()
fig.suptitle("Continnum model Bernevig with SOC")
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

ax.set_xticks(np.linspace(0,1,6))
ax.set_xticklabels(path_name)
plt.show()
print("TBG ended")




"""
#huj huj huj
            for k in range(0,8):
                #spin_z
                modul_1 = norm_2(vectors[k,j])
                eig_modul_1.append(modul_1)
            spin_up = sum(eig_modul_1)
            #print(spin_up)

            for k in range(8, 16):
                #spin_z
                modul_2 = norm_2(vectors[k,j])
                eig_modul_2.append(modul_2)
            spin_down = sum(eig_modul_2)
            #print(spin_down)
            for eigenvector in vectors[:,j]:
                #spin_z
                modul_1 = norm_2(eigenvector)
                eig_modul_1.append(modul_1)
                #modul_2 = norm_2(vectors[k+8])
                #eig_modul_2.append(modul_2)

            for eigenvector in vectors[:,j]:
                #spin_z
                modul_2 = norm_2(eigenvector)
                eig_modul_2.append(modul_2)

            spin_up = sum(eig_modul_1)
            print("spin_up: ",spin_up)
            spin_down = sum(eig_modul_2)
            print("spin_down: ",spin_down)
            norm = spin_up+spin_down
            print(norm)
            #expectation value of
            spin_tot = spin_up - spin_down

            #if spin_tot > 0:
            #    spin_tot = 1
            #if spin_tot <0:
            #    spin_tot = -1
            if spin_down > spin_up:
                print('DOWN')
                spin_tot = ((-1)*(spin_down - spin_up))
                #spin_tot = (-1)*(spin_down - spin_up)
                #spin_tot = (spin_up - spin_down)

                print(spin_tot)
            else:
                print("UP")
                #spin_tot = (spin_up - spin_down)
                spin_tot = (spin_down - spin_up)
                print(spin_tot)
"""           