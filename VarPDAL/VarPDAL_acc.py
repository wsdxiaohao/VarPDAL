#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 23:04:23 2023

@author: shidawang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 11:14:16 2023

@author: shidawang
"""

import time as clock
from genMetric import genMetric
from mymathtools import *
from mymath2 import *
def norm(v):
    #l2 norm of vector
    return np.sqrt(v.dot(v))
######################################
#      PDHG with line search
def VarPDAL_acc(model,oracle,options,tol,maxiter,check):
    #varaibel metric with line search
    """
    Variable metric Line search PDHG algorithm for solving
        min_{x}max_{y} <Dx,y> + KL(b,Ax) - \deta_{||.||2.1\leq \mu}(y) 
    subject to x>epsilon
    
        
    Dual step:
         y_k = prox_{\tau_km1} f^*(y_km1 + sig_km1 K x^k)
    Primal step:
        line search:
        sig_k <= sqrt(1+theta_km1) sig_km1, tau_k = beta sig_k, theta_k =sig_k/sig_km1    
        \bar y_k  = y_k + \theta_k (x_k - x_km1)
        x_kp1   = prox^M_{tau_k g} (x^k - tau_kM^-1\bar y_k - tau_kM^-1\nabla h (x_k) )
        backtracking condition:
            sig_k = sig_k * ratio
   
    
    model        model data of the optimization problem
        
    oracle:
    -'grad_h'   computes the gradient of the objective grad h(x^{k})
    -'prox_g'   computes the proximal mapping of g
    -'fun_g'    computes the value of g
    -'fun_h'    computes the value of h
    -'residual'  usded for breaking condition or resor plots
    
    options (required):
    -'init'     initialization
    
    options (optional):
    
    tol         tolerance threshold for the residual
    maxiter     maximal number of iterations
    check       provide information after 'check' iterations
    
    Return:
    -------
    output:
    -'sol'      solution of the problems
    -'seq_res'  sequence of residual values (if activiated)
    -'seq_time' sequence of time points (if activiated)
    -'seq_x'    sequence of iterates (if activiated)
    -'seq_obj'  sequence of objective values (if activiated)
    -'seq_beta' sequence of beta values ( )
    -'breakvalues' code for the type of breaking condition
                   1: maximal number of iterations exceeded
                   2: breaking condition reached (residual below tol)
                   3: not enough backtracking iteration
                   
    """
    
     # store options
    if 'storeResidual'  not in options:
        options['storeResidual']  = False;
    if 'storeTime'      not in options:
        options['storeTime']      = False;
    if 'storePoints'    not in options:
        options['storePoints']    = False;
    if 'storeObjective' not in options:
        options['storeObjective'] = False;
    if 'storeBeta' not in options:
        options['storeBeta'] = False;
        
    breakvalue = 0 ;
    # load oracle
    obj    = oracle['obj'];
    grad   = oracle['grad'];
    prox_g   = oracle['prox_g'];
    prox_fstar = oracle['prox_fstar']
    residual = oracle['residual'];
    PrimalSmooth = oracle['PrimalSmooth']
    #load parameter
    K = model['K']
    N = model['N']
    #initialization
    x_k = options['init_x'];
    x_kp1 =x_k.copy();
    
    y_k = options['init_y'];
    y_kp1 = y_k.copy();
    y_km1 = y_k.copy();
    sig_km1 = options['stepsize']
    sig_k = sig_km1
    theta_k = options['theta']
    beta = options['beta']
    delta =options['delta']
    epsilon = options['epsilon']
    m = options['memory']
    gamma = options['strong convexity']
    C = options['metric bound']
    
    
    objective = obj(model,x_k)
    res0 = residual( model, options,x_k);
    h_kp1 = PrimalSmooth(model,x_kp1)
    h_k = h_kp1
    grad_k = grad(model, options,x_k);
    grad_kp1 = grad_k
    
    
    parameter = {'sig':sig_k}
    
    
    if options['method'] =='mSR1':
        a0 = 0*np.ones(m*2)
    if options['method'] =='mBFGS':
        a0 = 0*np.ones(m*4)
    # B_k0
    B_k0 = 1 *np.ones(len(x_k))
    
    
    if options['storeResidual'] == True:
        seq_res = np.zeros(maxiter+1);
        seq_res[0] = 1;
    if options['storeTime'] == True:
        seq_time = np.zeros(maxiter+1);
        seq_time[0] = 0;
    if options['storePoints'] == True:
        seq_x = np.zeros((model['N'],maxiter+1));        
        seq_x[:,0] = x_kp1;
    if options['storeObjective'] == True:
        seq_obj = np.zeros(maxiter+1);        
        seq_obj[0] = objective;
    if options['storeBeta'] == True:
        seq_beta = np.zeros(maxiter);        
    time = 0;
    ListS = [];
    ListY = [];
    # solve 
    for iter in range(1,maxiter+1):
        stime = clock.time();
        # update varaible
        x_k = x_kp1
        y_km1 = y_k
        sig_km1 = sig_k
        theta_km1 = theta_k
        # compute gradient
        grad_k = grad_kp1
        #compute smooth term in the primal 
        h_k = h_kp1
        
        
     ########################## Dual step ###############################
        
        y_temp = y_km1 +sig_km1 * K.dot(x_k)
        
         
        y_k = prox_fstar(model, options, y_temp)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    ########################## line search ################################
        if options['line_search']==True:
            ratio =1+ gamma*beta*sig_km1/C
            beta = beta/(  ratio)
            #select initial sig_k and do backtracking
            sig_k = ratio* np.sqrt((1+theta_km1))*sig_km1;
            #sig_k = ratio * sig_km1
            linesearch_iter = 50
        #sig_k = sig_km1
        else:
            sig_k = sig_k
            linesearch_iter = 1
        
        
        
        for i in range(1,linesearch_iter+1):
            theta_k = sig_k/sig_km1
            tau_k = beta*sig_k
            options['stepsize'] = tau_k
           
            ybar_k = y_k + theta_k *(y_k-y_km1);
            
            
            
            ######################### Proximal Mapping ###################### 
            if iter<m+1:
                x_temp = x_k - tau_k*K.T.dot( ybar_k )- tau_k*grad_k
                x_kp1 = prox_g(model,tau_k,x_temp,epsilon)
            else:
                
                U1 = Metric['U1']
                U2 = Metric['U2']
                
                B0 = Metric['B0'];
               
                invB0 = (1/B0)
                m = Metric['memory'];
                if options['method'] == 'mSR1':
                        
                    I = np.identity(m)
                if options['method'] == 'mBFGS':
                    I = np.identity(2*m)
                
                #invB1 = invB0  - invB0.dot(U1.dot((np.linalg.inv(I + U1.T.dot(invB0.dot(U1))).dot(U1.T)).dot(invB0)))
                #invB2 = invB1 - invB1.dot(U2.dot((np.linalg.inv(I - U2.T.dot(invB1.dot(U2))).dot(-U2.T)).dot(invB1)))
                
                
                
                x_shift = tau_k*(K.T.dot(ybar_k)+grad_k)
                x_shift1 = shift1(I,B0,U1,x_shift)
                
                
                
                x_shift2 = shift2(I,B0,U1,U2,x_shift1)
                #invB2*x_shift
                #print('stepsize tau_k=')
                #print(options['stepsize'])
                
                
                ###### test #########
                #test_x = B0*x_shift2 + U1.dot(U1.T.dot(x_shift2)) - U2.dot(U2.T.dot(x_shift2))
                #testd = test_x - x_shift
                #product = (testd.dot(testd))
                #if (testd.dot(testd))>0.000001:
                #    print('sth rong')
                ##################
                
                
                
                x_temp = x_k - x_shift2 #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Where the problem lies.  
                #x_kp1 = x_temp
                #ak = a0
                x_kp1, ak = proxM_g(x_temp, a0, options, model, Metric)
                #a0 = ak
               
            
            
            
            
            
            
            #################################################################
            
            #smooth part
            h_kp1 = PrimalSmooth(model,x_kp1)
            #criterion of line 
            diff = x_kp1-x_k
            if iter<(m+1):
                norm_sq = (diff ).dot((diff )) 
            else:
                
                vector1 = U1.T.dot(diff)
                vector2 = U2.T.dot(diff)
                
                norm_sq1 = (diff ).dot((B0*diff )) 
                norm_sq2 = norm_sq1 + vector1.T.dot(vector1) - vector2.T.dot(vector2)
                norm_sq = norm_sq2#min(norm_sq1,norm_sq2)
            print('norm_sq%f'%norm_sq)
            temp = 2*tau_k*(h_kp1-h_k-grad_k.dot(diff))
            cri = tau_k*sig_k*(K.dot(diff)).dot(K.dot(diff))+ temp
            print('cri')
            print(cri)
            if(cri<delta*norm_sq):
                print('num of iteration of line search=%d'%i)
                
                break
            else:
                sig_k= 0.5*sig_k
            if (i == linesearch_iter)&(i>1):
                breakvalue = 1
                
        grad_kp1 = grad(model, options,x_kp1);
        if breakvalue == 1:
            
            print('line search is ended before the condition is satisfied')
            break
        
        
        #enerate metric for the next iteration
        #taping
        S_kp1 = x_kp1 - x_k
        Y_kp1 = grad_kp1 - grad_k
        SY = S_kp1.dot(Y_kp1)
        #BBstep = SY/S_kp1.dot(S_kp1)
        #print('******BBstep = %f****'%BBstep)
        
        
        print('SY')
        print(SY)
        if SY< 0:
            print('WARNING SY<0')
            break
        
        
            
            
        if iter<(m+1):
             ListS.append(S_kp1);
             ListY.append(Y_kp1);
             
        else:
            
            ListS[0:m-1] = ListS[1:m];
            ListY[0:m-1] = ListY[1:m];
            ListS[m-1] = S_kp1
            ListY[m-1] = Y_kp1
            
        
        if iter >= (m):
             S = np.stack(ListS,axis=-1)
             Y = np.stack(ListY,axis=-1)
             
           
             ###########################
             ###########################
             
             Stack = {'S_k':S, 'Y_k':Y, 'B_k0':B_k0,'memory':m}
             Metric = genMetric(Stack,options)
             
             
        
        
        
        
        
        
        
        # tape residual    
        res = residual( model, options,x_kp1);
        # tape time
        time = time + (clock.time() - stime);
        
        if options['storeTime'] == True:
            seq_time[iter] = time;
        if options['storePoints'] == True:
            seq_x[:,iter] = x_kp1;
        if options['storeObjective'] == True:
            seq_obj[iter] = h_kp1;
        if options['storeResidual'] == True:
            seq_res[iter] = res;

       
        
        
        
        # print info
        if (iter % check == 0):
            print ('iter: %d, time: %5f,res: %f' % (iter, time,res));
        
    # return results
    output = {
        'sol': x_kp1,
        'breakvalue': breakvalue
    }

    
    if options['storeTime'] == True:
        output['seq_time'] = seq_time;
    if options['storePoints'] == True:
        output['seq_x'] = seq_x;
    if options['storeObjective'] == True:
        output['seq_obj'] = seq_obj;
    if options['storeResidual'] == True:
        output['seq_res'] = seq_res;
    

    return output;
    

def shift1(I,B0,U1,x):
    invB0 = 1/B0
    return invB0*x - invB0* U1.dot(np.linalg.inv(I+U1.T.dot(invB0[:,np.newaxis]*U1)).dot(U1.T.dot(invB0*x)))
def shiftU(I,B0,U1,U2):
    invB0 = 1/B0
    #invB1U2
    #check
    
    return  (invB0[:,np.newaxis]*U2)- invB0[:,np.newaxis]*( U1.dot(np.linalg.inv(I+U1.T.dot(invB0[:,np.newaxis]*U1)).dot(U1.T.dot(invB0[:,np.newaxis]*U2))))
                               
                                   
def shift2(I,B0,U1,U2,x):
    Ushift = shiftU(I,B0,U1,U2)
    
    U_temp = U2.dot(np.linalg.inv(I - U2.T.dot(Ushift)).dot(U2.T.dot(x)))
    return  shift1(I,B0,U1,U_temp)+x
                                
    
def lm_primalstep(model,oracle,options,Metric,iteration,x_temp,a0):
    #limited_memory dual step
    #load parameters
    m = options['memory']
    prox_g = oracle['prox_g']
    i = iteration
    if i < (m+1):
        x_kp1 = prox_g(model, options, x_temp)
    else:
        
        #y_kp1 = prox_fstar(model,options,ybar_k)
        x_kp1 = proxM_g(x_temp,a0,options,model,Metric)
    return y_kp1
        
        
def eigmax(U, num_iterations=10):
    #power iteration tocal mx eigen only for UU^T matrix
    b_k = np.random.rand(U.shape[0])
    for _ in range(num_iterations):
        b_k1 = np.dot(U.T,b_k)
        b_k1 = np.dot(U,b_k1)
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1/b_k1_norm
    v = U.T.dot(b_k1) 
    rho = v.T.dot(v)/b_k.T.dot(b_k)
    return rho
        
        
        
        
        
        
        
        
    
        
        
        
        
        
        