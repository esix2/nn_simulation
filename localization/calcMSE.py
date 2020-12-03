#import numpy as np
#from numpy import exp, abs, angle
#import pandas as pd
#
#def calcMSE(Tx,Ty,Tx_hat,Ty_hat):
#    N, J = np.shape(Tx)
#    MSE = np.zeros((J,1),dtype=np.float)
#    for j in range(0,J):
#        z = np.array(Tx.transpose()[j])+1j*np.array(Ty.transpose()[j])
#        r1, theta1, polar = z2polar(z)
#        z_hat = np.array(Tx_hat.transpose()[j])+1j*np.array(Ty_hat.transpose()[j])
#        r2, theta2, polar_hat = z2polar(z_hat)
#        MSE_tmp = np.sqrt(r1**2+r2**2-2*r1*r2*np.cos(theta1-theta2))
#        MSE[j] = np.max(MSE_tmp)
#    return MSE
#def z2polar(z): 
#    N = np.size(z)
#    absZ = np.zeros((N,1),dtype=np.float)
#    polar_coord = np.zeros((N,2),dtype=np.float)
#    for n in range(0,N):
#        polar_coord[n][0] = abs(z[n])
#        polar_coord[n][1] = angle(z[n])
##    return absZ
#    polar_coord = pd.DataFrame(polar_coord,columns=['abs', 'ang'])
#    polar_coord = polar_coord.sort_values(by='abs')
##    polar_coord = polar_coord.values
#    angZ = polar_coord['ang'].values
#    absZ = polar_coord['abs'].values
#    return absZ, angZ, polar_coord

import numpy as np
from itertools import permutations
def calcMSE(Tx,Ty,Tx_hat,Ty_hat):
    N, J = np.shape(Tx)
    MSE = np.zeros((J,1),dtype=np.float)    
    idx = range(0,N)
    idx_perms = list(set(permutations(idx)))
    for j in range(0,J):
        MSE[j] = np.inf
        Tx_tmp = Tx[:,j]
        Ty_tmp = Ty[:,j]
        Tx_hat_tmp = Tx_hat[:,j]
        Ty_hat_tmp = Ty_hat[:,j]
        for i in range(0,len(idx_perms)):
            Tx_hat_permute = [Tx_hat_tmp[j] for j in idx_perms[i]]
            Ty_hat_permute = [Ty_hat_tmp[j] for j in idx_perms[i]]
            MSE_tmp = np.sqrt((Tx_tmp-Tx_hat_permute)**2+(Ty_tmp-Ty_hat_permute)**2)
            MSE[j] = min(max(MSE_tmp),MSE[j])
    return  MSE
