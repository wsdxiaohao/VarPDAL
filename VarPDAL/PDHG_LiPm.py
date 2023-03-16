#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 13:55:38 2023

@author: shidawang
"""

import time as clock
from genMetric import genMetric
from mymathtools import *
from mymath import *
def norm(v):
    #l2 norm of vector
    return np.sqrt(v.dot(v))
######################################
#      PDHG with line search
def PDHG_LiPm(model,oracle,options,tol,maxiter,check):
    #varaibel metric with line search
    """
    Line search PDHG algorithm for solving
        min_{x}max_{y} <Dx,y> + KL(b,Ax) - \deta_{||.||2.1\leq \mu}(y) 
    subject to x>epsilon
    
        
    Dual step:
         y_k = prox_{\tau_km1} f^*(y_km1 + sig_km1 K x^k)
    Primal step:
        line search:
        sig_k <= sqrt(1+theta_km1) sig_km1, tau_k = beta sig_k, theta_k =sig_k/sig_km1    
        \bar y_k  = y_k + \theta_k (x_k - x_km1)
        x_kp1   = prox_{tau_k g} (x^k - tau_k\bar y_k - tau_k\nabla h (x_k) )
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
        
    breakvalue = 1 ;
    # load oracle
    obj    = oracle['obj'];
    grad   = oracle['grad'];
    prox_g   = oracle['prox_g'];
    prox_fstar = oracle['prox_fstar']
    residual = oracle['residual'];
    PrimalSmooth = oracle['PrimalSmooth']
    
    line_search = options['line_search']
    #load parameter
    K = model['K']
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

    
    
    
    objective = obj(model,x_k)
    res0 = residual( model, options,x_k);
    h_kp1 = PrimalSmooth(model,x_kp1)
    h_k = h_kp1
    grad_k = grad(model, options,x_k);
    grad_kp1 = grad_k
    
    
    parameter = {'sig':sig_k}
    
    
    
    
    
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
            #select initial sig_k and do backtracking
            sig_k = np.sqrt((1+theta_km1))*sig_km1;
        
            for i in range(1,20+1):
                theta_k = sig_k/sig_km1
                tau_k = beta*sig_k
                ybar_k = y_k + theta_k *(y_k-y_km1);
                x_temp = x_k - tau_k*K.T.dot( ybar_k )- tau_k*grad_k
                x_kp1 = prox_g(model,tau_k,x_temp,epsilon)
                
                
                #smooth part
                h_kp1 = PrimalSmooth(model,x_kp1)
                #criterion of line 
                diff = x_kp1-x_k
                norm_sq = (diff ).dot(diff )
                
                temp = 2*tau_k*(h_kp1-h_k-grad_k.dot(diff))
                cri = tau_k*sig_k*(K.dot(diff)).dot(K.dot(diff))+ temp
                
                
                
                if(cri<delta*norm_sq):
                    print('num of iteration of line search=%d'%i)
                    print('norm_sq=%f'%norm_sq)
                    print('sigma=%f'%sig_k)
                    break
                else:
                    sig_k= 0.5*sig_k
            if i == maxiter:
                print('line search is ended before the condition is satisfied')
        else:
            tau_k = beta*sig_k
            ybar_k = y_k + theta_k *(y_k-y_km1);
            x_temp = x_k - tau_k*K.T.dot( ybar_k )- tau_k*grad_k
            x_kp1 = prox_g(model,tau_k,x_temp,epsilon)
                
        grad_kp1 = grad(model, options,x_kp1);
        
        S_kp1 = x_kp1 - x_k
        Y_kp1 = grad_kp1 - grad_k
        #SY = S_kp1.dot(Y_kp1)
        #BBstep = SY/S_kp1.dot(S_kp1)
        #print('******BBstep = %f****'%BBstep)
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
    
    
        
        
        
        
        
        
        
        
        
        
        
        
    
        
        
        
        
        
        