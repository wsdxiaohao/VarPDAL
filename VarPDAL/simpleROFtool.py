# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a tool script file for ROF problem.
"""

#######################################################
#######################################################
######                                         ########
######   min_x  0.5|u-b|_2^2 + \mu|Ku|_{2,1}   ########
######                                         ########
#######################################################
######  ROF model for image denoisying ################
import numpy as np
#computing Totalvariation norm
def TVnorm(n,K,x):
    #input: K is the matrix correspoding to derivative operator
    #       n is the length of x
    #       x is the vector of image
    N=2*n #len of Kx
    Kx=K.dot(x)
    Dx_2=(Kx*(Kx));# entrywise squared Kx
    Dx_tv=np.sum(np.sqrt(Dx_2[0:n]+Dx_2[n:N])); #Total variation of Dx
    return Dx_tv


def cal_primal_dual_gap_ROF(K,b,x,p,mu):
    #computing primal and dual gap for ROF When A=identity
    #input: K matrix of derivative2D
    #       b is the noisy image
    #       x is the generated image
    #       p is the dual variable
    #       mu is the parameter before regularization term
    n=b.shape[0]
    res=x-b;
    Dx_tv=TVnorm(n,K,x);
    primal=0.5*(res).dot(res)+mu*Dx_tv; 
    dual=-0.5*(K.T.dot(p)-b).dot(K.T.dot(p)-b)+0.5*b.dot(b);
    gap =primal-dual
    return gap


import numpy.linalg as lin
#to project vector onto a ball with radius mu    primal = 0.5*(res).dot(res)+mu*Dx;

def prox_proj_ball(p,mu):
    #input: p is a 1-D array 
    #       p[:n] for p_x part p[n:N] for p_y part
    #       mu is the regularization parameter
    N=int(len(p))
    n=int((N+1)/2)
    p_proj=np.zeros(N)
    p_norm=np.sqrt(p[:n]**2+p[n:N]**2)
    denominate=np.maximum(p_norm/mu,1)
    p_proj[:n]=p[:n]/denominate
    p_proj[n:N]=p[n:N]/denominate
    return p_proj


################################################################################################
################################# PDHG primal and dual hybrid descent method ###################
def pd( K, b, mu, tol, maxiter, check,tau,sig):#tau ,sig will be used when dealing with prox map
    #input: K derivative2D
    #       b noisy image;   mu regularization parameter ; maxiter maximum number of iterations
    #       check check points; tau sig
  
    #initialize the starting point
    x_kp1 = np.zeros(b.shape[0]);
    p_kp1 = np.zeros(2*b.shape[0])
    # solve 
    time = 0;
    breakvalue = 0;
    for iter in range(1,maxiter+1):
        #update x #########################################
        x_k=x_kp1.copy();
        p_k=p_kp1.copy();
        #########################################
        #checking breaking condition
        gap=cal_primal_dual_gap_ROF(K,b,x_k,p_k,mu)
        if (iter % check == 0):
            print ('iter: %d, gap: %f' % (iter, gap));
        if (gap < tol):
            breakvalue = 1;
            break;
        
        #########################################
        #prox map for x_k
        x_kp1=(tau*(b-K.T.dot(p_k))+x_k)/(1+tau)
        #########################################
        #prox map for p_k
        #p_temp save the intermediate data 
        p_temp=p_k+sig*(K.dot(2*x_kp1-x_k))#2*x_kp1-x_k
        
        #project (p_temp) onto ball of radius mu
        p_kp1=prox_proj_ball(p_temp,mu)
        
        #########################################
        
    output = {
        'primal_sol': x_kp1,
        'dual_sol': p_kp1,
        'breakvalue': breakvalue
    }
    
    return output


