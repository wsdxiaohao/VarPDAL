#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 10:46:44 2022

@author: shida
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

def Model():
    """
    Define the model data of the problem to be solved in this project.
    
    Returns:
    -------
    struct
    'A'   M x N matrix
    'b'   M vector
    'M'   Matrix dimension 1
    'N'   Matrix dimension 2
    'mu'  positive regularization weight
    'B'   defines the group structure (list of start
    and end-indizes of coordinates in the same group)
    """
    filename = "Diana240";
    img = mpimg.imread("data/" + filename + ".png"); 
    img = rgb2gray(img);
    (ny,nx) = np.shape(img);
    print ("image dimensions: ", np.shape(img))
    N = nx*ny
    oimg = img.reshape((N,1))
    ### construction of blurr kernel ###

    # filter of size 2*k+1
    k = 10;
    s = 2*k+1;
    filter = np.zeros((s,s));
    if False:   # construct Gaussian kernel
        sigma = 25;

        [dx,dy] = np.meshgrid(np.linspace(-k,k,s), np.linspace(-k,k,s));
        filter = np.exp(-(dx**2+dy**2)/(2.0*sigma**2));
        filter = filter/np.sum(filter);
        print(filter)
        plt.imshow(filter);
        plt.show()
    
    
    if True: # read filter from image
        filter_img = mpimg.imread("data/filter.png");
        s = np.shape(filter_img)[0];
        filter = filter_img/np.sum(filter_img);
    
    
    # blurr operator as matrix
    A = make_filter2D(ny, nx, filter); 
    
    N = nx*ny;
    # reshape to fit dimension of optimization variable
    imgN=sp_noise(img,0.1)
    b = imgN.reshape((N,1)); 
    b = img.reshape((N,1)); 

    #b=b+np.random.normal(0.0, 0.05, (N,1))Charkiw
    b = A.dot(b) + np.random.normal(0.0, 0.05, (N,1));
    #b=sp_noise(b,0.1)

    b = b.flatten();
    # write blurry image
    mpimg.imsave(filename + "blurry.png", b.reshape(nx,ny), cmap=plt.cm.gray);
    
    ### Model ######################################################################

    K = make_derivatives2D(ny, nx);
    #mu = 0.1;
    n = nx*ny
    D=K.dot(oimg[:,0])
    N = len(D)
    n = int(N/2)
    D[0:n] = np.sqrt( D[0:n]**2+D[n:N]**2)
    D[n:N] = D[0:n]
    beta = 1
    W = np.exp(-beta*D)
    
    
    ### Parameters
    tau=0.1
    sig=0.1
    mu=0.001
    model={'K':K,'A':A,'b':b,'mu':mu,'W':W}
    Metric={'Tau':tau,'Sigma':sig,'K':K}
    x=np.ones(K.shape[1])
    p=np.ones(K.shape[0])
    z={'x':x,'p':p}
    
    return model
    
    

def norm(v):
    return np.sqrt(v.dot(v))

### auxiliary functions
def pmax(A,B):
    return np.maximum(A,B);

def pmin(A,B):
    return np.minimum(A,B);


##############################################################
    
def prox_groupl2l1(x0,r,struct):
    B = struct['B'];
    x = x0.copy();
    x_sq = (r*x)**2
    for k in range(0,len(B)-1):
        dnrm = sqrt(sum( x_sq[B[k]:B[k+1]]));#**2
        if (dnrm<=1.0):
            x[B[k]:B[k+1]] = 0.0;
        else:
            x[B[k]:B[k+1]] = x[B[k]:B[k+1]] - x[B[k]:B[k+1]]/dnrm; #using moreau envelope
    return x;

def prox_l1(x0,r,struct):
    x = x0.copy();
    
    x = np.sign(x)*np.maximum(np.abs(x)-1/r,0)
    return x

def prox_fstarD(x0,r,model,Metric):
    ####### prox^D_(1/r)g ###########
    
    # prox^D_{(1/r) g}(x) =prox_{(1/(Dr)) g}(x) =x - 1/rD prox_{Drg^*}(Drx) 
    # D works as a scala since D is always assumed to be scaled identity (namely, D*Id) 
    ################################
    D = Metric['B0'];
    #B = struct['B'];
    x = x0.copy();
    x = prox_proj_ball(model,{},x)
    return x
        


def L_function(y0,a0,r, model, Metric):
    # L: R^2m->R^2m
    # load parameters
    U1 = Metric['U1'];
    U2 = Metric['U2'];
    D = Metric['B0'];
    #B = struct['B'];
    m = int(len(a0)/2)
    #D1 = D + U1xU1;
    #D1^{-1} = 
    # initialization
    y = y0.copy()
    a = a0.copy()
    
    y_temp = y + 1/(r)*(U2.dot(a[m:]))
    
    L1 = a[0:m]   
    L2 = U2.T.dot(y-   prox_fstarD((y_temp),r,model,Metric)   ) +a[m:]
    L = np.concatenate((L1,L2))
    return L;


def root_finding(y0,a_init,options,model,Metric):
    # Semi-smooth Newton method
    # inpute y0
    #load data
    Maxiter = 200;
    tol = 0.00000000001;
    m = Metric['memory'];
    a0 = a_init.copy();
    tau = options['stepsize'];
    
    
    mu = model['mu'];
    D = Metric['B0'];
    #
    d = D[0]
    U1 = Metric['U1'];
    U2 = Metric['U2'];
    r = 1.0/(tau)
    r = r/d
    for i in range(Maxiter):    
        z = y0 + 1/(r)*(U2.dot(a0[m:]))
        G = grad_l(z,r,model,Metric)
        la = L_function(y0,a0,r, model, Metric)
        a0 = a0 - np.linalg.inv(G).dot(la)
        #test
        print('******iteration=%f*******',i)
        print('norm(la)=',np.sqrt(la.dot(la)),'a=',a0)
        
        if ((la).dot(la))<tol:
            return a0
    print('not founded',a0)
    return a0
        

def prox_l1M(x0,a0,options,model,Metric):
    #compute proximal operator with respect to M
    #load data
    xk = x0.copy()
    a = a0.copy()
    D = Metric['B0'];
    U1 = Metric['U1'];
    U2 = Metric['U2'];
    invB0 = np.diag(1/D)
    
    invB1 = 1/D# replaced by using sherman morrison fomula
    
    m = Metric['memory'];
    tau = options['stepsize']
    mu = model['mu']
    r = 1.0/(tau)
    d = D[0]
    r = r/d
    #compute a
    a = root_finding(xk,a,options,model,Metric)
    #test 
    la =L_function(xk,a,r, model, Metric)
    #print('*****a =%.8f*****'%np.max(np.abs(a)))
    #print('*****la=%.8f*****'%norm(la))

    #compute shifted x

    x_temp = xk + invB1.dot(U2.dot(a[m:2*m]))
    x_kp1 = prox_fstarD(x_temp,r,model,Metric)
    
    return x_kp1,a
    
    
def grad_l(x0,r,model,Metric):
    #compute of \partial l:= [U1, U2]^T p(x) [B0^{-1}U1, -B1^{-1}] + [I, U1^TB1^{-1}U2 ]
    #                                                                [0,        I      ]
    #where $P(x)$ refers to the derivative of prox_groupl2
    #inpute x, struct, Metric
    #output G(x):=\partial l
    #load dataprox_groupl2
    U1 = Metric['U1'];
    U2 = Metric['U2'];
    D = Metric['B0'];
    
    m = Metric['memory'];
    mu = model['mu']
    N = model['N']
    n = 2*N
    x = x0.copy()
    invB0 =(1/D)
    invB1 =1/D# sherman morrison formula
    
    U = np.concatenate((U1,U2),axis=1)
    
    V = np.concatenate((invB0[:,np.newaxis]*(U1), -invB1[:,np.newaxis]*(U2)),axis=1)
    
    
    Q = np.identity(2*m)
    #Q[0:m,m:2*m] = U1.T.dot( invB1[:,np.newaxis]*(U2))
    DP , v=  grad_l1(x,r,model,Metric) ######### to correct
    
    #sparseMatrix
    #row = np.array([[i, i, i+N,i+N] for i in range(N)]).flatten()
    #col = np.array([[i,i+N,i,i+N] for i in range(N)]).flatten()
    #data = np.array([[v[i,0]*v[i,0],v[i,0]*v[i+N,0],v[i,0]*v[i+N,0],v[i+N,0]*v[i+N,0]] for i in range(N)]).flatten()
    #spM = csc_matrix((data, (row, col)), shape = (2*N, 2*N))
    #G =  U.T.dot(DP[:,np.newaxis]*V) +Q - (U.T.dot(spM.dot(V)))
    #G = U.T.dot(np.diag(DP).dot(V)) +Q
    G = Q + 1/r*U.T.dot(DP[:,np.newaxis]*V) #- 1/r* ( (U[:N,:]).T).dot(v[:,np.newaxis]*V[N:n,:])- 1/r* ( (U[N:n,:].T).dot(v[:,np.newaxis]*V[:N,:]))
    return G
    
def grad_l1(x0,mu,model,Metric):
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
    """
    #generate grad
    p_norm=np.sqrt(x[:N]**2+x[N:n]**2)
    
    
    DP=np.ones(n);
    v = np.ones((n,1))
    
    
    DP[:N] = mu*(p_norm>(mu))/(p_norm+tol )+DP[:N]*(p_norm<=(mu))
    DP[N:n] = mu*(p_norm>(mu))/(p_norm +tol )+DP[N:n]*(p_norm<=(mu))
    s = np.sqrt(mu)
    
    v[:N,0] = s*x[:N]*(p_norm>mu)/(p_norm+tol)**3  
    v[N:n,0] = s*x[N:n]*(p_norm>mu)/(p_norm+tol)**3  #### problem is here 
    """
    pnorm=np.sqrt(x[:N]**2+x[N:n]**2)
    p_norm = np.append(pnorm,pnorm)
    DP = mu*(p_norm>(mu))/(p_norm+tol )+ (p_norm<=(mu))
    v = (pnorm>mu)*((x[:N]*x[N:n])/(pnorm+(pnorm<=mu))**3)
    
    
    return DP,v

    
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

        
    
    
    
    
    
    
    
    