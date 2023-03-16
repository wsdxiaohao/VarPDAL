#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 18:29:29 2023

@author: shidawang
"""

from myimgtools import *

import numpy as np

from scipy.sparse import csc_matrix
from numpy.random import rand, uniform
from numpy.random import normal as randn
from numpy import abs, sum, max, sign,sqrt, maximum
from numpy import zeros, ones
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    
    output = np.zeros(image.shape)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.random()
            
            if rdn < prob:
                output[i][j] = 0
            else:
                output[i][j] = image[i][j]
    return output


    
    

def norm(v):
    return np.sqrt(v.dot(v))

### auxiliary functions
def pmax(A,B):
    return np.maximum(A,B);

def pmin(A,B):
    return np.minimum(A,B);

def shift1(I,B0,U1,x):
    invB0 = 1/B0;
    return invB0*x - invB0* U1.dot(np.linalg.inv(I+U1.T.dot(invB0[:,np.newaxis]*U1)).dot(U1.T.dot(invB0*x)))
def shiftU(I,B0,U1,U2):
    invB0 = 1/B0
    #invB1U2
    return  (invB0[:,np.newaxis]*U2) - invB0[:,np.newaxis]* U1.dot(np.linalg.inv(I+U1.T.dot(invB0[:,np.newaxis]*U1)).dot(U1.T.dot(invB0[:,np.newaxis]*U2)))
                               
                                   
##############################################################
def prox_orthant(model,tau,x0,epsilon):
    #x_i>epsilon
    x = x0.copy();
    x = (x>epsilon)*x + (x<=epsilon)*epsilon
    x = (x < 255)*x + (x >= 255)*255
    return x
def proxD_g(x0,tau,model,Metric,options):
    ####### prox^D_(1/r)g ###########
    
    
    ################################
    D = Metric['B0'];
    epsilon = options['epsilon']
    #B = struct['B'];
    x = x0.copy();
    x = prox_orthant(model,tau,x,epsilon)
    return x
        


def L_function(x0,a0,tau, model, Metric,options,I):
    # L: R^2m->R^2m
    # load parameters
    U1 = Metric['U1'];
    U2 = Metric['U2'];
    B0 = Metric['B0'];
    #B = struct['B'];
    m = int(len(a0)/2)
    #D1 = D + U1xU1;
    #D1^{-1} = 
    
    #initialization
    x = x0.copy()
    a = a0.copy()
    x_temp = x + shift1(I,B0,U1,  U2.dot(a[m:])) 
    #x_temp = x + invB1.dot(U2.dot(a[m:]))
    
    L1 = U1.T.dot(x_temp - proxD_g((x_temp- 1/B0 *U1.dot(a[0:m])),tau,model,Metric,options)) +a[0:m]   
    L2 = U2.T.dot(x-   proxD_g((x_temp- 1/B0 *U1.dot(a[0:m])),tau,model,Metric,options)   ) +a[m:]
    L = np.concatenate((L1,L2))
    return L;


def root_finding(y0,a_init,options,model,Metric,I):
    # Semi-smooth Newton method
    # inpute y0
    #load data
    Maxiter = 100;
    tol = 0.00000000001;
    m = Metric['memory'];
    a0 = a_init.copy();
    tau = options['stepsize'];
    
    
    mu = model['mu'];
    B0 = Metric['B0'];
    #
    
    U1 = Metric['U1'];
    U2 = Metric['U2'];
    
    
    if options['method'] == 'mSR1':
        
        m = m
    if options['method'] == 'mBFGS':
        m = 2*m
    for i in range(Maxiter):    
        z = y0 + shift1(I,B0,U1,(U2.dot(a0[m:])))- 1/B0 *U1.dot(a0[0:m])
        
        G = grad_l(z,tau,model,Metric,options,I)
        la = L_function(y0,a0,tau, model, Metric,options,I)
        a0 = a0 - np.linalg.inv(G).dot(la)
        #test
        #print('******iteration=%f*******',i)
        #print('norm(la)=',np.sqrt(la.dot(la)),'a=',a0)
        
        if ((la).dot(la))<tol:
            return a0
    print('not founded',a0)
    return a0
        



def proxM_g(x0,a0,options,model,Metric):
    #compute proximal operator with respect to M
    #load data
    xk = x0.copy()
    a = a0.copy()
    B0= Metric['B0'];
    invB0 =1/B0;
    U1 = Metric['U1'];
    U2 = Metric['U2'];
    
    m = Metric['memory'];
    n = len(B0)
    #invB1 = np.linalg.inv( np.diag(D) + U1.dot(U1.T))# replaced by using sherman morrison fomula
    if options['method'] == 'mSR1':
        
        I = np.identity(m)
    if options['method'] == 'mBFGS':
        I = np.identity(2*m)
    
    
   
    tau = options['stepsize']
    mu = model['mu']
    r = 1.0/(tau*mu)
    #compute a
    a = root_finding(xk,a,options,model,Metric, I)
    #test 
    la =L_function(xk,a,tau, model, Metric, options,I)
    #print('*****a =%.8f*****'%np.max(np.abs(a)))
    #print('*****la=%.8f*****'%norm(la))

    #compute shifted x
    if options['method'] == 'mSR1':
        x_temp = xk + shift1(I,B0,U1,(U2.dot(a[m:2*m]))) - invB0*(U1.dot(a[0:m]))
        x_kp1 = proxD_g(x_temp,tau,model,Metric,options)
    if options['method'] == 'mBFGS':
        x_temp = xk + shift1(I,B0,U1,(U2.dot(a[2*m:4*m]))) - invB0*(U1.dot(a[0:2*m]))
        x_kp1 = proxD_g(x_temp,tau,model,Metric,options)
    return x_kp1,a
    
    
def grad_l(x0,tau,model,Metric,options,I):
    #compute of \partial l:= [U1, U2]^T p(x) [B0^{-1}U1, -B1^{-1}] + [I, U1^TB1^{-1}U2 ]
    #                                                                [0,        I      ]
    #where $P(x)$ refers to the derivative of prox_groupl2
    #inpute x, struct, Metric
    #output G(x):=\partial l
    #load dataprox_groupl2
    U1 = Metric['U1'];
    U2 = Metric['U2'];
    B0 = Metric['B0'];
    
    m = Metric['memory'];
    mu = model['mu']
    N = model['N']
    n = 2*N
    x = x0.copy()
    invB0 = 1/B0
    
    
    U = np.concatenate((U1,U2),axis=1)
    
    V = np.concatenate((invB0[:,np.newaxis]*(U1), -shiftU(I,B0,U1,U2)),axis=1)
    
    
    
    
    if options['method'] == 'mSR1':
        Q = np.identity(2*m)
        invB1U2 = shiftU(I, B0, U1, U2)
        Q[0:m,m:2*m] = U1.T.dot( invB1U2)
        DP =  grad_orthant(x,tau,model,Metric,options)
        G = U.T.dot(DP[:,np.newaxis]*V) + Q
    if options['method'] == 'mBFGS':
        Q = np.identity(4*m)
        invB1U2 = shiftU(I, B0, U1, U2)
        Q[0:2*m,2*m:4*m] = U1.T.dot( invB1U2)
        DP =  grad_orthant(x,tau,model,Metric,options)
        G = U.T.dot(DP[:,np.newaxis]*V) + Q
    return G
    
def grad_orthant(x0,tau,model,Metric,options):
    #grad prox_{B}(x)
    #P = DP(diagonal part) -qq^T
    #output DP,q
    #load data
    tol = 0.00000000001;
    #B = struct['B']
    mu = model['mu']
    D = Metric['B0']
    N = model['N']
    x = x0.copy();
    n = 2*N;
    epsilon = options['epsilon']
    DP = (x0>epsilon)
    
    return DP

    
def prox_proj_ball(model,options,y):#p is a 1-D array p[:n] for p_x part p[n:N] for p_y part
    mu = model['mu']
    N=model['N']
    n=2*N
    p_proj=np.zeros(n)
    p_norm=np.sqrt(y[:N]**2+y[N:n]**2)
    p_norm = p_norm/mu
    denominate = np.maximum(p_norm,1)
    p_proj[:N]=y[:N]/denominate
    p_proj[N:n]=y[N:n]/denominate
    return p_proj
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

        
    
    
    
    
    
    
    
    