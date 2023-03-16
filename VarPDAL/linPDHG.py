#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 16:53:17 2022

@author: shidawang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 14:41:15 2022

@author: shidawang
"""

#PDHG + limited memory on the dual + backttracking method.
import numpy as np
import time as clock
from genMetric import genMetric
from mymathtools import *
from mymath import *
def norm(v):
    #l2 norm of vector
    return np.sqrt(v.dot(v))
def linPDHG(model,oracle,options,tol,maxiter,check):
    """
    Multimemory SR1 quasi-Newton PDHG algorithm for solving
        min_{x}max_{y} <Dx,y> + \frac{1}{2}||x-b||^2 - \deta_{||.||2.1\leq \mu}(y) 
        - \frac{1}{2}||W^{-1}y||^2
    g(x) =  \frac{1}{2}||x-b||^2 
    f^*(y ) = \deta_{||.||2.1\leq \mu}(y) 
    F(y)  = \frac{1}{2}||W^{-1}y||^2
        
    Primal step:
         x_k = prox_{\tau_km1} g(x_km1- tau_km1 K^* y^k)
    dual step:
        line search:
            
        \bar x  = x_km1 + \theta_k (x_k - x_km1)
        y_kp1   = prox^M_{sig_k f^*} (y^k + sig_k(\bar x - \nabla F (y) )
        backtracking
    
    Properties:
    -----------
        -'f'        convex, continuously differentiable with L-Lipschitz
        -'g'        convex, simple
    -----------
        mode        model data of the optimization problem
        
    oracle:
    -'grad_f'   computes the gradient of the objective grad f(x^{k})
    -'prox_g'   computes the proximal mapping of g
    -'fun_g'    computes the value of g
    -'fun_f'    computes the value of f
    -'residual'  usded for breaking condition or resor plots
    -'genMetric' used for generating Metric according to multi-memory SR1 quasi Newton method *****
    -'proxM_g'   computes the proximal mapping of g with respect to M
    options (required):
    -'stepsize' stepsize alpha = ?????  
    -'init'     initialization
    -'number of memory' m
    
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
    dualSmooth = oracle['dualSmooth']
    #load parameter
    K = model['K']
    
    #initialization
    x_k = options['init_x'];
    x_kp1 =x_k.copy();
    
    y_k = options['init_y'];
    y_kp1 = y_k.copy();
    y_km1 = y_k.copy();
    tau_k = options['stepsize']
    tau_kp1 = tau_k
    theta_k = options['theta']
    beta = options['beta']
    delta =options['delta']
    m = options['memory']
    objective = obj(model,x_k)
    res0 = residual( model, options,x_k);
    
    if options['method'] =='mSR1':
        a0 = np.ones(m*2)
    if options['method'] =='mBFGS':
        a0 = np.ones(m*4)
    # B_k0
    B_k0 = 5*np.ones(len(y_k))
    
    
    
    
    h_kp1 = dualSmooth(model,y_kp1)
    h_k = h_kp1
    grad_k = grad(model, options,y_k);
    grad_km1 = grad_k
    parameter = {'tau':tau_k}
    ############################
    # taping
    ListS = [];
    ListY = [];
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
        x_km1 = x_k
        y_km1 = y_k
        y_k = y_kp1
        tau_km1 = tau_k
        tau_k = tau_kp1
        theta_km1 = theta_k
        # compute gradient
        grad_km1 = grad_k
        grad_k = grad(model, options,y_kp1);
        #compute smooth term in the dual problem
        h_k = h_kp1
        
        #taping
        S_kp1 = y_k - y_km1
        Y_kp1 = grad_k - grad_km1
        print(S_kp1)
        if iter<(m+2):
            # check if S_kp1 and Y_kp1 OK
            #######
            if iter > 1:
                #######
                ListS.append(S_kp1)
                ListY.append(Y_kp1)
        else:
            print(ListS)
            #testSY = S_kp1.T.dot(B_k0.dot(S_kp1) - Y_kp1)
            if 1: #np.abs(testSY)>0:
                
                ListS[0:m-1] = ListS[1:m];
                ListY[0:m-1] = ListY[1:m];
                ListS[m-1] = S_kp1
                ListY[m-1] = Y_kp1
                
                
            else:
                print('something wrong!!!!!')
                break
        
        if iter >= (m+2):
             S = np.stack(ListS,axis=-1)
             Y = np.stack(ListY,axis=-1)
             
            
             Stack = {'S_k':S, 'Y_k':Y, 'B_k0':B_k0,'memory':m}
             Metric = genMetric(Stack,options)
        
        
        
        
        
        ########################## primal step ###############################
        
        x_temp = x_km1 - tau_km1*K.T.dot(y_k)
        parameter['tau']  = tau_km1
        x_k = prox_g(model,parameter,x_temp)
        
        
        res = residual( model, options,x_k);
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        ########################## line search ################################
        #choose tau_k = [tau_km1, tau_km1*sqrt((1+theta_km1))]
        
        
        
        
        
        if options['line_search'] == False:
            theta_k = 1
            
            sig_k = beta*tau_k
            xbar_k = x_k + theta_k *(x_k-x_km1)
            ybar_k = y_k + sig_k*(K.dot(xbar_k)- grad_k)
            y_kp1 = prox_fstar(model, options, ybar_k)
            
            #smooth part
            h_kp1 = dualSmooth(model,y_kp1)
        if options['line_search'] == True:
            tau_k = tau_km1*np.sqrt(1+theta_km1)
            
            for i in range(1,maxiter+1):
                theta_k = tau_k/tau_km1;
                #theta_k = 1
                sig_k = beta*tau_k
                options['stepsize'] = sig_k
                xbar_k = x_k + theta_k *(x_k-x_km1)
                #ybar_k = y_k + sig_k*(K.dot(xbar_k)- grad_k)
                y_shift = sig_k*(K.dot(xbar_k)- grad_k)
                
                ##########################################################
                #                       dual                             #
                
               
                ybar_k = y_k + sig_k*(K.dot(xbar_k)- grad_k)
                
                y_kp1 = prox_fstar(model, options, ybar_k)
                
                ##########################################################
            
                #smooth part
                h_kp1 = dualSmooth(model,y_kp1)
                #criterion of line 
                diff = y_kp1-y_k
                norm_sq = (diff ).dot(diff )
            
                temp = 2*sig_k*(h_kp1-h_k-grad_k.dot(diff))
                cri = tau_k*sig_k*(K.T.dot(diff)).dot(K.T.dot(diff))+ temp
            
                if(cri<delta*norm_sq):
                    print('num of iteration of line search=%d'%i)
                    break
                else:
                    tau_kp1= 0.5*tau_k
                if i == maxiter:
                    print('line search is ended before the condition is satisfied')
            
                
        # tape residual   
        res = residual( model, options,x_k);
        # tape time
        time = time + (clock.time() - stime);
        # print info
        if (iter % check == 0):
            print ('iter: %d, time: %5f,res: %f' % (iter, time,res));
        
    
        
        
        

    
    return 0


    
def lm_dualstep(model,oracle,options,Metric,iteration,ybar_k,a0):
    #limited_memory dual step
    #load parameters
    m = options['memory']
    prox_fstar = oracle['prox_fstar']
    i = iteration
    if i < (m+1):
        y_kp1 = prox_fstar(model, options, ybar_k)
    else:
        
        #y_kp1 = prox_fstar(model,options,ybar_k)
        y_kp1 = prox_l1M(ybar_k,a0,options,model,Metric)
    return y_kp1
    
    
    