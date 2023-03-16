#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 16:30:39 2022

@author: shida
"""

import numpy as np
from numpy import zeros, sqrt
import time as clock

from mymath import *
def genMetric(Stack,optional):
    """
    genMetric for generating Metric according to multi-memory quasi-Newton method
    
    Construction:
        S_k^T Y_k
        D_k = D(S_k^T Y_k)
        L_k = L(S_k^T Y_k)
        
        A_k = [B_{k,0} Y_k]  -----------------------------BFG
        Q_k = [ -S_k^T B_{k,0} S_k , -L_k]----------------BFG
              [ -L^T_k,               D_k]
              
        
        A_k = Y_k - B_{k,0}S_k----------------------------SR1
        Q_k = D_k + L_k +L^T_K - S_k^T B_{k,0} S_k -------SR1
        
        Q_k = V^T E V spectral decompostion
        E = E1 - E2   where E1: positive spectrals, E2:= np.abs(negative spectrals)
        
        U1 = (A_kV^T)E_1^{1/2}
        U2 = (A_kV^T)E_2^{1/2}
        
    Variables:
    -----------
    s^k := x^{k+1} - x^k
    y^k := grad_f(x^{k+1}) - grad_f(x^{k})
    
    -----------
    Stack:
    -'S_k'     S_k := [s^{k-m}...s^{k-1}]
    -'Y_k'     Y_k := [y^{k-m}...y^{k-1}]
    -'B_k0'     B_{k,0}
    -'memory' m
        
    Method (required):  'L-BFG' or 'mSR1'
    
    
    output:
    -'B0'
    -'U1'
    -'U2'
    """
    # load data
    S_k = Stack['S_k'];
    Y_k = Stack['Y_k'];
    B_k0 = Stack['B_k0'];
    m = Stack['memory'];
    
    
    
    #initialization
    alpha = 0.01
    #useful variables
    SY  = np.dot(S_k.T,Y_k);
    B_k0 = B_k0
    D_k = np.diag(SY); #diagonal
    L_k = np.tril(SY,k=-1); #lower trin Pointer shows you how to set up PyTorch on a cloud-based environment, then walks you through the creation of neural architectures that facilitate operations on images, sound, text, and more through deep dives into each element. He also covers the critical concepts of applying transfer learning to images, debugging models, and PyTorch in production.angle
    if optional['method'] =='mSR1':
        # let Q_k always positive
        A_k = Y_k - B_k0[:,np.newaxis]*S_k
        #A_k = Y_k - np.dot(np.diag(B_k0), S_k)
        Q_k = np.diag(D_k) + L_k + L_k.T - np.dot(S_k.T,    B_k0[:,np.newaxis]*S_k   )
        
        #Q_k =SY - np.dot(S_k.T,  B_k0[:,np.newaxis]*S_k    ) 
        #inv Q_k
        """
        if np.max(np.abs(Q_k))<0.0000001:
            U_1 = np.zeros((len(B_k0),m))
            U_2 = np.zeros((len(B_k0),m))
            print('Q_k not invertible')
            output={'B0':B_k0,
                   'U1':U_1, #test
                   'U2':U_2,
                   'memory':m,
                   'Q_k':Q_k
                   }
            return output
        """
    if optional['method'] == 'mBFGS':
        #BS = (np.diag(B_k0)).dot(S_k)
        BS = B_k0[:,np.newaxis]*(S_k)
        #Ak
        A_k = np.concatenate((BS , Y_k) , axis =1)
        #Qk
        Q_k = np.zeros((2*m,2*m))
        Q_k[:m,:m] = -(S_k.T).dot(BS)
        Q_k[:m,m:] = - L_k
        Q_k[m:,:m] = - L_k.T
        Q_k[m:,m:] = np.diag(D_k)
        
        
        
    
    Eig, V = np.linalg.eig(np.linalg.inv(Q_k)) #decomposition of Q_k, where V is orthogonal.
        #Remark: We have to ensure Eig >0
        ##### test ####
        #print ('to check decomposition is correct', V.T.dot(V))
        ##############
    print('######## eig=')
    print(Eig)
    Eig = (Eig)
    Eig_pos = (Eig>0)*np.sqrt(np.abs(Eig))
    if np.max(Eig>0):
        print('***** positive eigenvalue ******')
    Eig_neg = (Eig<0)*np.sqrt(np.abs(Eig))
    ###############
    
    U_1 = 1*(A_k.dot(V))*Eig_pos
    U_2 = 1*(A_k.dot(V))*Eig_neg
   
    
    
    #test if Mk largest eigenvalue bounded or not
    d1 = np.diag(U_1.T.dot(U_1))
    d2 = np.diag(U_2.T.dot(U_2))
    eigenvalue_bound = np.max(d1)+np.max(d2)+np.max(B_k0)

    bound = optional['metric bound'] -alpha
    if (eigenvalue_bound<bound):
        pass
    else:
        U_1 = U_1 * bound/eigenvalue_bound
        U_2 = U_2 * bound/eigenvalue_bound
        
    
        
    
    
    
    
    
    
    
    output={'B0':B_k0+alpha,
           'U1':U_1, 
           'U2':U_2,
           'memory':m,
           'Q_k':Q_k
            }
    
    #print('U1 and U2', np.max(np.abs(U_1)),np.max(np.abs(U_2)))
    return output
    
    
    