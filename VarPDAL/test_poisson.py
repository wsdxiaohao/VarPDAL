#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 21:40:02 2023

@author: shidawang
"""

from mymathtools import * #(waiting for editing)
import scipy as sp
#################################################################
### Image deblurring problem under poison noise ###############################################
#                                                               #
# Optimization problem:                                         #
#                                                               #
#   min_x KL(b,Ax) + ||Dx||_{2,1}                               #
#  subject to x_{i,j} > epsilon                                 #
#where                                                          #
#   D  spatial finite difference                                #
#   KL(b,Ax) sum_{i,j} Ax_{i,j} - b_{i,j} log Ax_{i,j} neg      #
#Model:                                                         #
#   A     is a blurring operator                                #
#   b     is the blurry image                                   #
#   mu    positive parameter                                    #
#   N     dimension of the optimization variable     N =nx x ny #
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
    mpimg.imsave(filename + "ground_truth.png", img, cmap=plt.cm.gray);
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
    A = make_filter2D(ny, nx, filter); 
    #A = 1
    N = nx*ny;
    
    b = oimg
    np.random.seed(0)
    PEAK = 0.001
    noisy = np.random.poisson(b / 255.0 * PEAK) / PEAK / 255
    b = A.dot(b) + noisy;
    #b=sp_noise(b,0.1)

    b = b.flatten();
    # write blurry image
    mpimg.imsave(filename + "Poisson-blurry.png", b.reshape(nx,ny), cmap=plt.cm.gray);
    
    # K
    K = make_derivatives2D(ny, nx);
    
    
    # mu
    mu = mu_init
    
    model = {'K':K,'A':A,'b':b,'mu':mu, 'N':N,'nx':nx,'ny':ny}
    return model
    
    
    
###############################################################################
np.random.seed(0)
filename = 'Diana240'
model = Model("Diana240",0.0001)  


compute_optimal_value = False;



########## Define problem specific oracles ####################################
########### zero oder oracle ##################################################
def objectiveSmooth(model,x_init):
    #the KL divergence
    #load parameters
    x = x_init
    K = model['K']
    A = model['A']
    b = model['b']
    mu = model['mu']
    N = model['N']
   
    #    
    Ax = A.dot(x)
    KL = np.sum(Ax - b*np.log(Ax))
    return KL
def objectiveNonSmooth(model,x_init):
    #the KL divergence
    #load parameters
    x = x_init
    K = model['K']
    A = model['A']
    mu = model['mu']
    N = model['N']
    
    Kx = K.dot(x)
    Dx_2 = (Kx*(Kx));# entrywise squared Kx
    TV = np.sum(np.sqrt(Dx_2[0:N]+Dx_2[N:2*N])); #Total variation of Dx
    
    return TV

def objective(model, x_init):
    #load parameters
    x = x_init.copy();
    mu = model['mu'];
    #
    KL = objectiveSmooth(model, x)
    TV = objectiveNonSmooth(model,x);
    obj = KL + mu*TV;
    return obj

###############################################################################
####### proximal map oracle ###################################################
def prox_orthant(model,tau,x0,epsilon):
    #x_i>epsilon
    x = x0.copy();
    x = (x>epsilon)*x + (x<=epsilon)*epsilon
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
def grad_Smooth(model,options,x0):
    # load parameter
    A = model['A'];
    b = model['b'];
    x = x0.copy();
    grad = 1-b/(A.dot(x));
    grad = A.T.dot(grad)
    return grad
##############################################################################
if compute_optimal_value:
    def residual(model,options,x):
        return objective(model,x);
else:
    h_sol = np.load('data_poisson.npy');
    def residual(model, options,x):
        
        return objective(model,x) - h_sol
###############################################################################
# load algorithm

from PDHG_LiPm import PDHG_LiPm
from PDHG_VaLiPm import PDHG_VaLiPm
from PrLiPm import PDHG_PrLiPm
##################################################################
#general parameter
maxiter = 1000;
check = 10;
tol = 0.0000001;
bound = 50; #we set an upper bound of M_k
# UZILLLIey vARIABLES
N = model['N']
# initialization
x0 =1*np.ones(model['N']);
y0 = 1*np.ones(2*model['N'])
#x0 = np.ones(model['N'])/model['N'];
#Lipschitz constant
A = model['A']
K = model['K']
Lip = sp.sparse.linalg.norm(A.T.dot(A))
LipK = 1 #p.sparse.linalg.norm(K.T.dot(K))

sig0 = 1/(5*LipK+ Lip) #0.2/(LipK+ Lip)

beta = 5;

# taping:
xs = [];
rs = [];
ts = [];
cols = [];
legs = [];
nams = [];


# turn algorithms to be run on or off
run_PDHG = 0;
run_PDHG_LiPm = 0;
run_PDHG_VaLiPm = 1;
run_PDHG_PrLiPm = 0
if compute_optimal_value: # optimal solution is compyted using FISTA
    maxiter = 10000;
    check = 1;
    run_PDHG_LiPm = 1;
    run_PDHG_VaLiPm = 0;
    run_PDHG_PrLiPm = 0;
    run_PDHG = 0;
#####################################################################
#Lip =
#print('Lip=%f'%Lip)
"""
#####################################################################
if run_PDHG:
    
    print('');
    print('********************************************************');
    print('***PDHG Line search on the Primal varaible***');
    print('***********');
    
    options = {
        'init_x':          x0,
        'init_y':          y0,
        'theta':          1.0,
        'beta':            beta,
        'delta':           0.9,
        'epsilon':         1e-10,
        'storeResidual': True,
        'storeTime':     True,   
        'stepsize':       sig0,
        'storePoints':True,
    
        'storeObjective':True,
        'storeBeta':True,
        'line_search':  False,
        
    }

    oracle = {
        'obj':   objective,
        'objNonSm':    objectiveNonSmooth,
        'grad':   grad_Smooth,        
        'prox_g':   prox_orthant,
        'prox_fstar':  prox_proj_ball,
        'residual': residual,
        'PrimalSmooth': objectiveSmooth
    }
    
    output = PDHG_LiPm(model,oracle,options,tol, maxiter,check);
    
    xs.append(output['sol']);
    rs.append(output['seq_res']);
    ts.append(output['seq_time']);
    cols.append((0,0,0.5,1));
    legs.append('PDHG');
    nams.append('Lin');
    x = output['sol'];
    nx = model['nx']
    ny = model['ny']
    #if compute_optimal_value:
        #np.save('data_poisson.npy',rs[0][-1], allow_pickle=True, fix_imports=True)
    #mpimg.imsave(filename + "reconstruction(line search).png", x.reshape(ny,nx), cmap=plt.cm.gray);
############################################################################

#####################################################################
if run_PDHG_LiPm:
    
    print('');
    print('********************************************************');
    print('***PDHG Line search on the Primal varaible***');
    print('***********');
    
    options = {
        'init_x':          x0,
        'init_y':          y0,
        'theta':          1.0,
        'beta':            beta,
        'delta':           0.9,
        'epsilon':         1e-10,
        'storeResidual': True,
        'storeTime':     True,   
        'stepsize':       2,
        'storePoints':True,
    
        'storeObjective':True,
        'storeBeta':True,
        'line_search':  True,
        
    }

    oracle = {
        'obj':   objective,
        'objNonSm':    objectiveNonSmooth,
        'grad':   grad_Smooth,        
        'prox_g':   prox_orthant,
        'prox_fstar':  prox_proj_ball,
        'residual': residual,
        'PrimalSmooth': objectiveSmooth
    }
    
    output = PDHG_LiPm(model,oracle,options,tol, maxiter,check);
    
    xs.append(output['sol']);
    rs.append(output['seq_res']);
    ts.append(output['seq_time']);
    cols.append((0,0,1,1));
    legs.append('PDAL');
    nams.append('Lin');
    x = output['sol'];
    nx = model['nx']
    ny = model['ny']
    if compute_optimal_value:
        np.save('data_poisson.npy',rs[0][-1], allow_pickle=True, fix_imports=True)
    mpimg.imsave(filename + "reconstruction(line search).png", x.reshape(ny,nx), cmap=plt.cm.gray);
############################################################################

"""
if run_PDHG_VaLiPm:
    m =1
    print('');
    print('********************************************************');
    print('***PDHGVarible Line search on the Primal varaible***');
    print('***********');
    
    options = {
        'init_x':          x0,
        'init_y':          y0,
        'theta':          1.0,
        'beta':            1,
        'delta':           0.9,
        'epsilon':         1e-10,
        'storeResidual': True,
        'storeTime':     True,   
        'stepsize':       sig0,
        'storePoints':True,
        'memory':      m,
        'storeObjective':True,
        'storeBeta':True,
        'line_search':  False,
        'method':      'mBFGS',
        'metric bound': bound,
    }

    oracle = {
        'obj':   objective,
        'objNonSm':    objectiveNonSmooth,
        'grad':   grad_Smooth,        
        'prox_g':   prox_orthant,
        'prox_fstar':  prox_proj_ball,
        'residual': residual,
        'PrimalSmooth': objectiveSmooth
    }
    
    output = PDHG_VaLiPm(model,oracle,options,tol, maxiter,check);
    
    xs.append(output['sol']);
    rs.append(output['seq_res']);
    ts.append(output['seq_time']);
    cols.append((1,0.0,0.5,1));
    legs.append('VarPDHG memory=%d'%m);
    nams.append('Lin');
    x = output['sol'];
    nx = model['nx']
    ny = model['ny']
    
    #mpimg.imsave(filename + "reconstruction(variable metric line search).png", x.reshape(ny,nx), cmap=plt.cm.gray);
############################################################################
"""
"""
############################################################################


if run_PDHG_VaLiPm:
    m =5
    print('');
    print('********************************************************');
    print('***PDHGVarible Line search on the Primal varaible***');
    print('***********');
    
    options = {
        'init_x':          x0,
        'init_y':          y0,
        'theta':          1.0,
        'beta':            1,
        'delta':           0.9,
        'epsilon':         1e-10,
        'storeResidual': True,
        'storeTime':     True,   
        'stepsize':       sig0,
        'storePoints':True,
        'memory':      m,
        'storeObjective':True,
        'storeBeta':True,
        'line_search':  False,
        'method':      'mBFGS',
        'metric bound': bound,
    }

    oracle = {
        'obj':   objective,
        'objNonSm':    objectiveNonSmooth,
        'grad':   grad_Smooth,        
        'prox_g':   prox_orthant,
        'prox_fstar':  prox_proj_ball,
        'residual': residual,
        'PrimalSmooth': objectiveSmooth
    }
    
    output = PDHG_VaLiPm(model,oracle,options,tol, maxiter,check);
    
    xs.append(output['sol']);
    rs.append(output['seq_res']);
    ts.append(output['seq_time']);
    cols.append((1,0.0,0,1));
    legs.append('VarPDHG memory=%d'%m);
    nams.append('Lin');
    x = output['sol'];
    nx = model['nx']
    ny = model['ny']
    
    #mpimg.imsave(filename + "reconstruction(variable metric line search).png", x.reshape(ny,nx), cmap=plt.cm.gray);
############################################################################

"""



if run_PDHG_VaLiPm:
    m =1
    print('');
    print('********************************************************');
    print('***PDHGVarible Line search on the Primal varaible***');
    print('***********');
    
    options = {
        'init_x':          x0,
        'init_y':          y0,
        'theta':          1.0,
        'beta':            beta,
        'delta':           0.9,
        'epsilon':         1e-10,
        'storeResidual': True,
        'storeTime':     True,   
        'stepsize':       1,
        'storePoints':True,
        'memory':      m,
        'storeObjective':True,
        'storeBeta':True,
        'line_search':  True,
        'method':      'mBFGS'
    }

    oracle = {
        'obj':   objective,
        'objNonSm':    objectiveNonSmooth,
        'grad':   grad_Smooth,        
        'prox_g':   prox_orthant,
        'prox_fstar':  prox_proj_ball,
        'residual': residual,
        'PrimalSmooth': objectiveSmooth
    }
    
    output = PDHG_VaLiPm(model,oracle,options,tol, maxiter,check);
    
    xs.append(output['sol']);
    rs.append(output['seq_res']);
    ts.append(output['seq_time']);
    cols.append((1,0.5,0,1));
    legs.append('VarPDAL memory=%d'%m);
    nams.append('Lin');
    x = output['sol'];
    nx = model['nx']
    ny = model['ny']
    
    #mpimg.imsave(filename + "reconstruction(variable metric line search).png", x.reshape(ny,nx), cmap=plt.cm.gray);


############################################################################
if run_PDHG_VaLiPm:
    m = 3
    
    print('');
    print('********************************************************');
    print('***PDHG Line search on the Primal varaible***');
    print('***********');
    
    options = {
        'init_x':          x0,
        'init_y':          y0,
        'theta':          1.0,
        'beta':            beta,
        'delta':           0.9,
        'epsilon':         1e-10,
        'storeResidual': True,
        'storeTime':     True,   
        'stepsize':       1,
        'storePoints':True,
        'memory':      m,
        'storeObjective':True,
        'storeBeta':True,
        'line_search':  True,
        'method':      'mBFGS'
    }

    oracle = {
        'obj':   objective,
        'objNonSm':    objectiveNonSmooth,
        'grad':   grad_Smooth,        
        'prox_g':   prox_orthant,
        'prox_fstar':  prox_proj_ball,
        'residual': residual,
        'PrimalSmooth': objectiveSmooth
    }
    
    output = PDHG_VaLiPm(model,oracle,options,tol, maxiter,check);
    
    xs.append(output['sol']);
    rs.append(output['seq_res']);
    ts.append(output['seq_time']);
    cols.append((0,0.5,0,1));
    legs.append('VarPDAL memory=%d'%m);
    nams.append('Lin');
    x = output['sol'];
    nx = model['nx']
    ny = model['ny']
    
    mpimg.imsave(filename + "reconstruction(variable metric line search).png", x.reshape(ny,nx), cmap=plt.cm.gray);
############################################################################


############################################################################
if run_PDHG_VaLiPm:
    m = 5
    print('');
    print('********************************************************');
    print('***PDHG Line search on the Primal varaible***');
    print('***********');
    
    options = {
        'init_x':          x0,
        'init_y':          y0,
        'theta':          1.0,
        'beta':            beta,
        'delta':           0.9,
        'epsilon':         1e-10,
        'storeResidual': True,
        'storeTime':     True,   
        'stepsize':       1,
        'storePoints':True,
        'memory':      m,
        'storeObjective':True,
        'storeBeta':True,
        'line_search':  True,
        'method':      'mBFGS'
    }

    oracle = {
        'obj':   objective,
        'objNonSm':    objectiveNonSmooth,
        'grad':   grad_Smooth,        
        'prox_g':   prox_orthant,
        'prox_fstar':  prox_proj_ball,
        'residual': residual,
        'PrimalSmooth': objectiveSmooth
    }
    
    output = PDHG_VaLiPm(model,oracle,options,tol, maxiter,check);
    
    xs.append(output['sol']);
    rs.append(output['seq_res']);
    ts.append(output['seq_time']);
    cols.append((0.75,0.25,1,1));
    legs.append('VarPDAL memory=%d'%m);
    nams.append('Lin');
    x = output['sol'];
    nx = model['nx']
    ny = model['ny']
    
    mpimg.imsave(filename + "reconstruction(variable metric line search).png", x.reshape(ny,nx), cmap=plt.cm.gray);
############################################################################




if run_PDHG_VaLiPm:
    m = 7
    print('');
    print('********************************************************');
    print('***PDHG Line search on the Primal varaible***');
    print('***********');
    
    options = {
        'init_x':          x0,
        'init_y':          y0,
        'theta':          1.0,
        'beta':            beta,
        'delta':           0.9,
        'epsilon':         1e-10,
        'storeResidual': True,
        'storeTime':     True,   
        'stepsize':       1,
        'storePoints':True,
        'memory':      m,
        'storeObjective':True,
        'storeBeta':True,
        'line_search':  True,
        'method':      'mBFGS'
    }

    oracle = {
        'obj':   objective,
        'objNonSm':    objectiveNonSmooth,
        'grad':   grad_Smooth,        
        'prox_g':   prox_orthant,
        'prox_fstar':  prox_proj_ball,
        'residual': residual,
        'PrimalSmooth': objectiveSmooth
    }
    
    output = PDHG_VaLiPm(model,oracle,options,tol, maxiter,check);
    
    xs.append(output['sol']);
    rs.append(output['seq_res']);
    ts.append(output['seq_time']);
    cols.append((0.5,0.25,0.25,1));
    legs.append('VarPDAL memory=%d'%m);
    nams.append('Lin');
    x = output['sol'];
    nx = model['nx']
    ny = model['ny']
    
    #mpimg.imsave(filedeblurring_name + "reconstruction(variable metric line search).png", x.reshape(ny,nx), cmap=plt.cm.gray);


############################################################################
if run_PDHG_VaLiPm:
    m = 9
    print('');
    print('********************************************************');
    print('***PDHG Line search on the Primal varaible***');
    print('***********');
    
    options = {
        'init_x':          x0,
        'init_y':          y0,
        'theta':          1.0,
        'beta':            beta,
        'delta':           0.9,
        'epsilon':         1e-10,
        'storeResidual': True,
        'storeTime':     True,   
        'stepsize':       1,
        'storePoints':True,
        'memory':      m,
        'storeObjective':True,
        'storeBeta':True,
        'line_search':  True,
        'method':      'mBFGS'
    }

    oracle = {
        'obj':   objective,
        'objNonSm':    objectiveNonSmooth,
        'grad':   grad_Smooth,        
        'prox_g':   prox_orthant,
        'prox_fstar':  prox_proj_ball,
        'residual': residual,
        'PrimalSmooth': objectiveSmooth
    }
    
    output = PDHG_VaLiPm(model,oracle,options,tol, maxiter,check);
    
    xs.append(output['sol']);
    rs.append(output['seq_res']);
    ts.append(output['seq_time']);
    cols.append((0.25,0.25,0.25,1));
    legs.append('VarPDAL memory=%d'%m);
    nams.append('Lin');
    x = output['sol'];
    nx = model['nx']
    ny = model['ny']
    
    #mpimg.imsave(filedeblurring_name + "reconstruction(variable metric line search).png", x.reshape(ny,nx), cmap=plt.cm.gray);
############################################################################

































"""
###############################################################################

if run_PDHG_PrLiPm:
    
    print('');
    print('********************************************************');
    print('***PDHG Line search on the Primal varaible***');
    print('***********');
    
    options = {
        'init_x':          x0,
        'init_y':          y0,
        'theta':          1.0,
        'beta':            1,
        'delta':           0.9,
        'epsilon':         0.0000001,
        'storeResidual': True,
        'storeTime':     True,   
        'stepsize':       1,
        'storePoints':True,
        'memory':      5,
        'storeObjective':True,
        'storeBeta':True,
        'line_search':  True,
        'method':      'mSR1',
        'metric bound': bound,
    }

    oracle = {
        'obj':   objective,
        'objNonSm':    objectiveNonSmooth,
        'grad':   grad_Smooth,   
        'prox_g':   prox_orthant,
        'prox_fstar':  prox_proj_ball,
        'residual': residual,
        'PrimalSmooth': objectiveSmooth
    }
    
    output = PDHG_PrLiPm(model,oracle,options,tol, maxiter,check);
    
    xs.append(output['sol']);
    rs.append(output['seq_res']);
    ts.append(output['seq_time']);
    cols.append((0,0.5,1,1));
    legs.append('Preconditioning ');
    nams.append('Lin');
    x = output['sol'];
    nx = model['nx']
    ny = model['ny']
    #mpimg.imsave(filename + "reconstruction(Preconditioning line search).png", x.reshape(ny,nx), cmap=plt.cm.gray);
############################################################################

"""










################################################################################
#########################################################################
nalgs = len(rs);

# plotting
fig1 = plt.figure();
iterations = np.arange(0,len(rs[0])-2,1)
for i in range(0,nalgs):
    #plt.plot(ts[i][1:-1], rs[i][1:-1], '-', color=cols[i], linewidth=2);
    plt.plot(iterations,rs[i][1:-1], '-', color=cols[i], linewidth=2);

plt.legend(legs);
plt.yscale('log');
plt.xscale('log');

plt.xlabel('iterations')
plt.ylabel('Primal gap');
plt.title('deblurring image under Poisson noise')
plt.savefig('deblurring image under Poisson noise BFGS New.pdf')
plt.show();

nalgs = len(rs);

# plotting
fig2 = plt.figure();
iterations = np.arange(0,len(rs[0])-2,1)
for i in range(0,nalgs):
    plt.plot(ts[i][1:-1], rs[i][1:-1], '-', color=cols[i], linewidth=2);
    #plt.plot(iterations,rs[i][1:-1], '-', color=cols[i], linewidth=2);

plt.legend(legs);
plt.yscale('log');
plt.xscale('log');

plt.xlabel('time')
plt.ylabel('Primal gap');
plt.title('deblurring image under Poisson noise')
plt.savefig('deblurring image under Poisson noise BFGS(time) New.pdf')
plt.show();



"""


###############################################################################

# taping:
xs = [];
rs = [];
ts = [];
cols = [];
legs = [];
nams = [];


# turn algorithms to be run on or off
run_PDHG = 1;
run_PDHG_LiPm = 1;
run_PDHG_VaLiPm = 1;
run_PDHG_PrLiPm = 0




#####################################################################
if run_PDHG:
    
    print('');
    print('********************************************************');
    print('***PDHG Line search on the Primal varaible***');
    print('***********');
    
    options = {
        'init_x':          x0,
        'init_y':          y0,
        'theta':          1.0,
        'beta':            beta,
        'delta':           0.9,
        'epsilon':         1e-10,
        'storeResidual': True,
        'storeTime':     True,   
        'stepsize':       sig0,
        'storePoints':True,
    
        'storeObjective':True,
        'storeBeta':True,
        'line_search':  False,
        'metric bound': bound,
        
    }

    oracle = {
        'obj':   objective,
        'objNonSm':    objectiveNonSmooth,
        'grad':   grad_Smooth,        
        'prox_g':   prox_orthant,
        'prox_fstar':  prox_proj_ball,
        'residual': residual,
        'PrimalSmooth': objectiveSmooth
    }
    
    output = PDHG_LiPm(model,oracle,options,tol, maxiter,check);
    
    xs.append(output['sol']);
    rs.append(output['seq_res']);
    ts.append(output['seq_time']);
    cols.append((0,0,0.5,1));
    legs.append('PDHG');
    nams.append('Lin');
    x = output['sol'];
    nx = model['nx']
    ny = model['ny']
    #if compute_optimal_value:
        #np.save('data_poisson.npy',rs[0][-1], allow_pickle=True, fix_imports=True)
    #mpimg.imsave(filename + "reconstruction(line search).png", x.reshape(ny,nx), cmap=plt.cm.gray);
############################################################################

#####################################################################
if run_PDHG_LiPm:
    
    print('');
    print('********************************************************');
    print('***PDHG Line search on the Primal varaible***');
    print('***********');
    
    options = {
        'init_x':          x0,
        'init_y':          y0,
        'theta':          1.0,
        'beta':            beta,
        'delta':           0.9,
        'epsilon':         1e-10,
        'storeResidual': True,
        'storeTime':     True,   
        'stepsize':       2,
        'storePoints':True,
    
        'storeObjective':True,
        'storeBeta':True,
        'line_search':  True,
        'metric bound': bound,
        
    }

    oracle = {
        'obj':   objective,
        'objNonSm':    objectiveNonSmooth,
        'grad':   grad_Smooth,        
        'prox_g':   prox_orthant,
        'prox_fstar':  prox_proj_ball,
        'residual': residual,
        'PrimalSmooth': objectiveSmooth
    }
    
    output = PDHG_LiPm(model,oracle,options,tol, maxiter,check);
    
    xs.append(output['sol']);
    rs.append(output['seq_res']);
    ts.append(output['seq_time']);
    cols.append((0,0,1,1));
    legs.append('PDAL');
    nams.append('Lin');
    x = output['sol'];
    nx = model['nx']
    ny = model['ny']
    if compute_optimal_value:
        np.save('data_poisson.npy',rs[0][-1], allow_pickle=True, fix_imports=True)
    mpimg.imsave(filename + "reconstruction(line search).png", x.reshape(ny,nx), cmap=plt.cm.gray);
############################################################################


if run_PDHG_VaLiPm:
    m =1
    print('');
    print('********************************************************');
    print('***PDHGVarible Line search on the Primal varaible***');
    print('***********');
    
    options = {
        'init_x':          x0,
        'init_y':          y0,
        'theta':          1.0,
        'beta':            1,
        'delta':           0.9,
        'epsilon':         1e-10,
        'storeResidual': True,
        'storeTime':     True,   
        'stepsize':       sig0,
        'storePoints':True,
        'memory':      m,
        'storeObjective':True,
        'storeBeta':True,
        'line_search':  False,
        'method':      'mBFGS',
        'metric bound': bound,
    }

    oracle = {
        'obj':   objective,
        'objNonSm':    objectiveNonSmooth,
        'grad':   grad_Smooth,        
        'prox_g':   prox_orthant,
        'prox_fstar':  prox_proj_ball,
        'residual': residual,
        'PrimalSmooth': objectiveSmooth
    }
    
    output = PDHG_VaLiPm(model,oracle,options,tol, maxiter,check);
    
    xs.append(output['sol']);
    rs.append(output['seq_res']);
    ts.append(output['seq_time']);
    cols.append((1,0.0,0.5,1));
    legs.append('VarPDHG memory=%d'%m);
    nams.append('Lin');
    x = output['sol'];
    nx = model['nx']
    ny = model['ny']
    
    #mpimg.imsave(filename + "reconstruction(variable metric line search).png", x.reshape(ny,nx), cmap=plt.cm.gray);
############################################################################

############################################################################


if run_PDHG_VaLiPm:
    m =5
    print('');
    print('********************************************************');
    print('***PDHGVarible Line search on the Primal varaible***');
    print('***********');
    
    options = {
        'init_x':          x0,
        'init_y':          y0,
        'theta':          1.0,
        'beta':            1,
        'delta':           0.9,
        'epsilon':         1e-10,
        'storeResidual': True,
        'storeTime':     True,   
        'stepsize':       sig0,
        'storePoints':True,
        'memory':      m,
        'storeObjective':True,
        'storeBeta':True,
        'line_search':  False,
        'method':      'mBFGS',
        'metric bound': bound,
    }

    oracle = {
        'obj':   objective,
        'objNonSm':    objectiveNonSmooth,
        'grad':   grad_Smooth,        
        'prox_g':   prox_orthant,
        'prox_fstar':  prox_proj_ball,
        'residual': residual,
        'PrimalSmooth': objectiveSmooth
    }
    
    output = PDHG_VaLiPm(model,oracle,options,tol, maxiter,check);
    
    xs.append(output['sol']);
    rs.append(output['seq_res']);
    ts.append(output['seq_time']);
    cols.append((1,0.0,0,1));
    legs.append('VarPDHG memory=%d'%m);
    nams.append('Lin');
    x = output['sol'];
    nx = model['nx']
    ny = model['ny']
    
    #mpimg.imsave(filename + "reconstruction(variable metric line search).png", x.reshape(ny,nx), cmap=plt.cm.gray);
############################################################################


############################################################################
if run_PDHG_VaLiPm:
    m = 9
    print('');
    print('********************************************************');
    print('***PDHG Line search on the Primal varaible***');
    print('***********');
    
    options = {
        'init_x':          x0,
        'init_y':          y0,
        'theta':          1.0,
        'beta':            beta,
        'delta':           0.9,
        'epsilon':         1e-10,
        'storeResidual': True,
        'storeTime':     True,   
        'stepsize':       1,
        'storePoints':True,
        'memory':      m,
        'storeObjective':True,
        'storeBeta':True,
        'line_search':  True,
        'method':      'mBFGS',
        'metric bound': bound,
    }

    oracle = {
        'obj':   objective,
        'objNonSm':    objectiveNonSmooth,
        'grad':   grad_Smooth,        
        'prox_g':   prox_orthant,
        'prox_fstar':  prox_proj_ball,
        'residual': residual,
        'PrimalSmooth': objectiveSmooth
    }
    
    output = PDHG_VaLiPm(model,oracle,options,tol, maxiter,check);
    
    xs.append(output['sol']);
    rs.append(output['seq_res']);
    ts.append(output['seq_time']);
    cols.append((0.25,0.25,0.25,1));
    legs.append('VarPDAL memory=%d'%m);
    nams.append('Lin');
    x = output['sol'];
    nx = model['nx']
    ny = model['ny']
    
    mpimg.imsave(filename + "reconstruction(variable metric line search).png", x.reshape(ny,nx), cmap=plt.cm.gray);
############################################################################


nalgs = len(rs);

# plotting
fig1 = plt.figure();
iterations = np.arange(0,len(rs[0])-2,1)
for i in range(0,nalgs):
    #plt.plot(ts[i][1:-1], rs[i][1:-1], '-', color=cols[i], linewidth=2);
    plt.plot(iterations,rs[i][1:-1], '-', color=cols[i], linewidth=2);

plt.legend(legs);
plt.yscale('log');
plt.xscale('log');

plt.xlabel('iterations')
plt.ylabel('Primal gap');
plt.title('deblurring image under Poisson noise')
plt.savefig('deblurring image under Poisson noise BFGS New Second.pdf')
plt.show();

nalgs = len(rs);

# plotting
fig2 = plt.figure();
iterations = np.arange(0,len(rs[0])-2,1)
for i in range(0,nalgs):
    plt.plot(ts[i][1:-1], rs[i][1:-1], '-', color=cols[i], linewidth=2);
    #plt.plot(iterations,rs[i][1:-1], '-', color=cols[i], linewidth=2);

plt.legend(legs);
plt.yscale('log');
plt.xscale('log');

plt.xlabel('time')
plt.ylabel('Primal gap');
plt.title('deblurring image under Poisson noise')
plt.savefig('deblurring image under Poisson noise BFGS(time) New Second.pdf')
plt.show();








