#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 10:34:57 2022

@author: shida
"""
from mymathtools import * #(waiting for editing)
#################################################################
### Lasso problem ###############################################
#                                                               #
# Optimization problem:                                         #
#                                                               #
#   min_x G(x) + f*h(Kx)                             #
#                                                               #
#where                                                          #
#   G(x) = 0.5|x-b|^2
#   f(x) = mu||Kx||_{2,1}                    #
#   h(x) = 1/2\epsilon ||Wx||_2^2
#   $W$ is a diagnal matrix $diag(w_1,w_2,...,w_n)$          #
#       which is designed to detect edges by assigning weights  #
#                       on the pixels along edges in the image. #                                                          #
#Model:                                                         #
#   A     MxN Matrix                                            #
#   b     M vector                                              #
#   mu    positive parameter                                    #
#   N     dimension of the optimization variable                #
#################################################################

def Model(filename, mu_init):
    """
    Define the model data of the problem to be solved in this project.
    
    Returns:
    -------
    struct
    'A'   the filter
    'b'   the blurred image
    
    'mu'  the positive regularization parameter
    'K'   the discrete 'differential' of the image.
    'W'   defines the weight in favor of the difference along edges
    """
    #filename = "Diana240";
    img = mpimg.imread("data/" + filename + ".png"); 
    img = rgb2gray(img);
    (ny,nx) = np.shape(img);
    print ("image dimensions: ", np.shape(img))
    
    #flatten image
    N = nx*ny
    oimg = img.reshape((N,1))
    
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
        plt.show();
    if True: # read filter from image
        filter_img = mpimg.imread("data/filter.png");
        s = np.shape(filter_img)[0];
        filter = filter_img/np.sum(filter_img);
    # blurr operator as matrix
    #A = make_filter2D(ny, nx, filter); 
    #A = 1
    N = nx*ny;
    # reshape to fit dimension of optimization variable
    imgN=sp_noise(img,0.1)
    b = imgN.reshape((N,1)); 
    b = img.reshape((N,1)); 

    #b=b+np.random.normal(0.0, 0.05, (N,1))Charkiw
    b = b + np.random.normal(0.0, 0.05, (N,1));
    #b=sp_noise(b,0.1)

    b = b.flatten();
    # write blurry image
    mpimg.imsave(filename + "blurry.png", b.reshape(nx,ny), cmap=plt.cm.gray);
    
    # K
    K = make_derivatives2D(ny, nx);
    
    # W
    D=K.dot(oimg[:,0])
    n = len(D)
    m = int(n/2)
    D[0:m] = np.sqrt( D[0:m]**2+D[m:n]**2)
    D[m:n] = D[0:m]
    beta = 1
    W = np.exp(-beta*D)
    # mu
    mu = mu_init
    
    model = {'K':K,'b':b,'mu':mu, 'W':W, 'N':N}
    return model
    
    
    
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
    
###############################################################################
model = Model("Diana240",0.01)  
compute_optimal_value = 1

########## Define problem specific oracles ####################################
########### zero oder oracle ##################################################
def objectiveNonSmooth(model,x_init):
    #the infimal convolution term
    #load parameters
    x = x_init
    K = model['K']
    mu = model['mu']
    N = model['N']
    W = model['W']
    #    
    kx = K.dot(x)
    temp = np.sqrt(kx[0:N]**2 +  kx[N:2*N]**2)
    w = W[0:N]**2/mu
    infimal_conv_norm = ((temp - 1/w)<=0).dot(0.5*w*temp**2) + ((temp - 1/w)>0).dot(temp-1/(2*w))
    return infimal_conv_norm

def objective(model, x_init):
    #load parameters
    x = x_init.copy();
    #A = model['A'];
    b = model['b'];
    mu = model['mu'];
    #
    r = x-b;
    Dx_infimal = objectiveNonSmooth(model,x);
    obj = 0.5*(r).dot(r)+mu*Dx_infimal; 
    return obj
def dualSmooth(model,y_init):
    y = y_init
    W = model['W']
    w=1/W
    return 0.5*(w*y).dot(w*y)
###############################################################################
####### proximal map oracle ###################################################
def prox_l2(model,parameter,x0):
    #0.5*|x-b|^2
    tau = parameter['tau'];
    b = model['b']
    
    d = 1/tau;
    x = prox_sql2(x0+b/d,d) 
    return x
#### check later ########
def prox_proj_ball(model,options,y):#p is a 1-D array p[:n] for p_x part p[n:N] for p_y part
    mu = model['mu']
    N=model['N']
    n=2*N
    p_proj=np.zeros(n)
    p_norm=np.sqrt(y[:N]**2+y[N:n]**2)
    p_norm=p_norm/mu
    denominate=np.maximum(p_norm,1)
    p_proj[:N]=y[:N]/denominate
    p_proj[N:n]=y[N:n]/denominate
    return p_proj

###############################################################################
######## first order oracle ###################################################
def grad_DualSmooth(model,options,y0):
    W = model['W']
    w = 1/W
    return w*y0
##############################################################################
if compute_optimal_value:
    def residual(model,options,x):
        return objective(model,x);
else:
    h_sol = np.load('data.npy');
    def residual(model, options,x):
        
        return objective(model,options,x) - hsol
###############################################################################

# load algorithm
from limited_memo_PDHG import limited_memo_PDHG
from linPDHG import linPDHG
##################################################################
#general parameter
maxiter = 300;
check = 20;
tol = -1;

# UZILLLIey vARIABLES
N = model['N']
# initialization
x0 =np.zeros(model['N']);
y0 = np.zeros(2*model['N'])
#x0 = np.ones(model['N'])/model['N'];

# taping:
xs = [];
rs = [];
ts = [];
cols = [];
legs = [];
nams = [];


# turn algorithms to be run on or off
run_zeroSR1 = 0 #zero SR1 Proximal Quasi-Newton
run_zeroBFGS = 0; #multi-memory SR1 Proximal Quasi-Newton
run_fista = 1; #FISTA
run_linPDHG= 1;
run_fbs = 1;
run_multiBFGS = 0;
if compute_optimal_value: # optimal solution is compyted using FISTA
    maxiter = 10;
    check = 1;
    run_lmPDHG = 1;
    run_linPDHG = 1;
#####################################################################


W = model['W']
w = 1/W
Lip= np.max(w*w)
print('Lip=%f'%Lip)
#####################################################################

if run_linPDHG:
    
    print('');
    print('********************************************************');
    print('***FBS***');
    print('***********');
    
    options = {
        'init_x':          x0,
        'init_y':          y0,
        'stepsize':      0.1,
        'theta':          1.0,
        'beta':            0.5,
        'delta':           0.99,
        'storeResidual': True,
        'storeTime':     True,   
        'number of memory': 3,
        'storePoints':True,
        'method': 'mSR1',
        'storeObjective':True,
        'storeBeta':True,
        'line_search':  True,
        'memory':       2
    }

    oracle = {
        'obj':   objective,
        'objNonSm':    objectiveNonSmooth,
        'grad':   grad_DualSmooth,        
        'prox_g':   prox_l2,
        'prox_fstar':  prox_proj_ball,
        'residual': residual,
        'dualSmooth': dualSmooth
    }
    
    output = linPDHG(model,oracle,options,tol, maxiter,check);
    #xs.append(output['sol']);
    #rs.append(output['seq_res']);
    #ts.append(output['seq_time']);
    #cols.append((1,0.95,0,1));
    #legs.append('fbs');
    #nams.append('fbs');
    print(output)
############################################################################




if run_lmPDHG:
    
    print('');
    print('********************************************************');
    print('***FBS***');
    print('***********');
    
    options = {
        'init_x':          x0,
        'init_y':          y0,
        'stepsize':      0.1,
        'theta':          1.0,
        'beta':            0.5,
        'delta':           0.99,
        'storeResidual': True,
        'storeTime':     True,   
        'number of memory': 3,
        'storePoints':True,
        'method': 'mSR1',
        'storeObjective':True,
        'storeBeta':True,
        'line_search':  True,
        'memory':       2
    }

    oracle = {
        'obj':   objective,
        'objNonSm':    objectiveNonSmooth,
        'grad':   grad_DualSmooth,        
        'prox_g':   prox_l2,
        'prox_fstar':  prox_proj_ball,
        'residual': residual,
        'dualSmooth': dualSmooth
    }
    
    output = limited_memo_PDHG(model,oracle,options,tol, maxiter,check);
    #xs.append(output['sol']);
    #rs.append(output['seq_res']);
    #ts.append(output['seq_time']);
    #cols.append((1,0.95,0,1));
    #legs.append('fbs');
    #nams.append('fbs');
    print(output)
############################################################################













 

















    