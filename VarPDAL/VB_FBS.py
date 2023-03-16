#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 15:30:19 2022

@author: shida
"""
import numpy as np
from numpy import zeros, sqrt
import time as clock
from genMetric import genMetric
from mymath import *
def VB_FBS(model, oracle, options, tol, maxiter, check):
    """
    Multimemory  quasi-Newton fast backtracking FBS algorithm for solving
        min_{x} h(x); h(x):= g(x) + f(x)
    initialization tau_k0 qk=mu tau_k0 / (1+tau_k0 mu_g)
    input mu mu_g, mu_f, rho, x_k=y_k=x_km1
    Update step:
        forwardstep:
                x_ktemp = y_k - tau * grad_g(y_k)
        backwardstep:
                backtracking:
                    if for tau_k0, CB2 holds:
                        i = 0
                        if CB1 is not satisfied and i<imax:
                            tau_k =rho^i * tau_k0
                            x_kp1   = prox^M_tau_k*g(x_ktemp)    where M = D+-Q
                            i = i+1
                    elseif for tau_k0 CB2 is not satisfied:
                        tau_k = tau_k0/rho
                        x_kp1   = prox^M_tau*g(x_ktemp)    where M = D+-Q
                set tau_kp10 = tau_k,
                    q_kp1 = mu tau_kp10 / (1+tau_kip10 mu_g)
                    computing t_kp1,
                    computing beta_kp1,
                    y_kp1 = x_k + beta_kp1 (x_k-x_km1)
                    
                
    
    
    
    Here:
        CB: Df(x_kp1,x_k)<= norm(x_kp1-x_k)^2/tau_k
        CB2: 2DF(x_kp1,x_k)/norm(x_kp1-x_k)^2 > rho *(1/tau_k)
    ###########################################################################
    
    ###########################################################################
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
#store options
    #backtracking options
    
    #load oracle
    fun_f = oracle['fun_f'];
    fun_g = oracle['fun_g'];
    grad_f = oracle['grad_f'];
    prox_g = oracle['prox_g'];
    residual = oracle['residual'];
    prox_M = oracle['prox_M'];
    #load parameter
    tau_k0   = options['stepsize'];
    tau_kp10 = tau_k0
    
    # initalization
    x_kp1 = options['init'];
    y_kp1 = x_kp1.copy()
    x_k   = x_kp1.copy()
    y_k   = x_kp1.copy()
    imax = 10
    rho = options['backtracking parameter']
    f_kp1 = fun_f(x_kp1, model, options);
    h_kp1 = f_kp1 + fun_g(x_kp1,model,options);
    res0  = residual(x_kp1, 1.0, model,options);
    B = model['B']
    m = options['number of memory']
     # B_k0
    B_k0 = 1*np.ones(len(x_k))
    # compute gradient
    grad_kp1 = grad_f(x_k, model, options);
    #
    if options['method'] =='mSR1':
        a0 = np.ones(m*2)
    if options['method'] =='mBFGS':
        a0 = np.ones(m*4)
    # taping
    ListS = []
    ListY = []
    # taping
    if options['storeResidual'] == True:
        seq_res = zeros(maxiter+1);
        seq_res[0] = 1;
    if options['storeTime'] == True:
        seq_time = zeros(maxiter+1);
        seq_time[0] = 0;
    if options['storePoints'] == True:
        seq_x = zeros((model['N'],maxiter+1));        
        seq_x[:,0] = x_kp1;
    if options['storeObjective'] == True:
        seq_obj = zeros(maxiter+1);        
        seq_obj[0] = h_kp1;
    if options['storeBeta'] == True:
        seq_beta = zeros(maxiter);        
    time = 0;
    # solve
    breakvalue = 1;
    for iter in range(1,maxiter+1):
        stime = clock.time();
        
        
        
        
        # update variables
        
        y_k = y_kp1.copy();
        
        f_k = f_kp1.copy();
        grad_k = grad_kp1.copy()
        
        tau_k0 = tau_kp10
        q_k =q_kp1
        
        #backward step
        # Here we are going to use multimemory SR1 (prox^M_g)
        if iter<(m+1):
            #forward step
            x_temp = y_k -tau * grad_k
            #
            if model['mu']>0:
                
            #x_kp1 = x_temp
                #starting backtracking #######################################
                
                
                
                
                x_kp1 = prox_g(x_temp,tau_k0,model,options)
                f_kp1 = fun_f(x_kp1)
                CB = checkCB(x_kp1,x_k,f_kp1,f_k,grad_k)
                norm = norm(x_kp1-x_k)
                if 2*CB/norm>(1/tau_k0):
                    while (CB>norm/(2*tau_k0)) &(i<imax):
                        tau_k = rho**i*tau_k0
                        x_kp1 = prox_g(x_temp,tau_k,model,options)
                        f_kp1 = fun_f(x_kp1)
                        i = i+1
                        CB = checkCB(x_kp1,x_k,f_kp1,f_k,grad_k)
                        norm = norm(x_kp1-x_k)
                else:
                    tau_k = tau_k0/rho
                    x_kp1 = prox_g(x_temp,tau_k,model,options)
                    
                
                
                ##############################################################
            else:
                x_kp1 = x_temp
        else:
            
            #forward step
            U1 = Metric['U1']
            U2 = Metric['U2']
            B0 = Metric['B0']
            Metric['B0'] = B0
            D = Metric['B0'];
            
            invB0 = np.diag(1/D)
            m = Metric['memory'];
            if options['method'] == 'mSR1':
                
                I = np.identity(m)
            if options['method'] == 'mBFGS':
                I = np.identity(2*m)
            
            
            #x_kp1 = prox_g(x_temp,tau,model,options)
            #x_kp1 = prox_M(x_temp,options,model,Metric)#,model)
            if model['mu']>0:
                
                #start backtracking on tau_k ##########################################
                
                
                
                invB1 = invB0 - invB0.dot(U1.dot((np.linalg.inv(I + U1.T.dot(invB0.dot(U1))).dot(U1.T)).dot(invB0)))
            
                invB2 = invB1 - invB1.dot(U2.dot((np.linalg.inv(I - U2.T.dot(invB1.dot(U2))).dot(-U2.T)).dot(invB1)))
                x_temp = y_k - tau_k0*invB2.dot(grad_k)
                
                
                
                x_kp1,a0 = prox_M(x_temp,a0,options,model,Metric)
                
                f_kp1 = fun_f(x_kp1)
                CB = checkCB(x_kp1,x_k,f_kp1,f_k,grad_k)
                norm = norm(x_kp1-x_k)
                
                if 2*CB/norm>(1/tau_k0):
                    while (CB>norm/(2*tau_k0)) &(i<imax):
                        tau_k = rho**i*tau_k0
                        invB1 = invB0 - invB0.dot(U1.dot((np.linalg.inv(I + U1.T.dot(invB0.dot(U1))).dot(U1.T)).dot(invB0)))
            
                        invB2 = invB1 - invB1.dot(U2.dot((np.linalg.inv(I - U2.T.dot(invB1.dot(U2))).dot(-U2.T)).dot(invB1)))
                        x_temp = y_k - tau_k*invB2.dot(grad_k)
                
                
                
                        x_kp1,a0 = prox_M(x_temp,a0,options,model,Metric)
                        
                        
                        
                        f_kp1 = fun_f(x_kp1)
                        i = i+1
                        CB = checkCB(x_kp1,x_k,f_kp1,f_k,grad_k)
                        norm = norm(x_kp1-x_k)
                else:
                    tau_k = tau_k0/rho
                    invB1 = invB0 - invB0.dot(U1.dot((np.linalg.inv(I + U1.T.dot(invB0.dot(U1))).dot(U1.T)).dot(invB0)))
            
                    invB2 = invB1 - invB1.dot(U2.dot((np.linalg.inv(I - U2.T.dot(invB1.dot(U2))).dot(-U2.T)).dot(invB1)))
                    x_temp = y_k - tau_k*invB2.dot(grad_k)
                
                
                
                    x_kp1,a0 = prox_M(x_temp,a0,options,model,Metric)
                    
                
                ######################################################
            else:
                x_kp1 = x_temp
            #x_kp1 = x_temp
        ########## extrapolation smooth ##################
        # cal t
        tau_kp10 = tau_k
        q_kp1 = (mu*tau_kp10)/(1+tau_kp10mu_g)
        t_kp1 = cal_t(t_k, q_k, q_kp1)
        
        #cal beta
        
        beta_kp1 = cal_beta(t_k, t_kp1, tau_kp10, mu_g, mu, mu_f)
        
        y_kp1 = x_kp1 + beta_kp1*(x_kp1- x_k)
        #########################################
        # compute gradient
        grad_kp1 = grad_f(y_kp1, model, options);
        # generate Metric
        # stack vectors
        S_kp1 = x_kp1 - x_k
        Y_kp1 = grad_kp1 - grad_k
        
        if iter<(m+1):
            # check if S_kp1 and Y_kp1 OK
            #######
            
            
            #######
            ListS.append(S_kp1)
            ListY.append(Y_kp1)
            
    
        else:
            testSY = S_kp1.T.dot(B_k0.dot(S_kp1) - Y_kp1)
            if np.abs(testSY)>0:
                
                ListS[0:m-1] = ListS[1:m];
                ListY[0:m-1] = ListY[1:m];
                ListS[m-1] = S_kp1
                ListY[m-1] = Y_kp1
            else:
                print('something wrong!!!!!')
                break
        if iter >= m:
            S = np.stack(ListS,axis=-1)
            Y = np.stack(ListY,axis=-1)
        
            Stack = {'S_k':S, 'Y_k':Y, 'B_k0':B_k0,'memory':m}
            Metric = genMetric(Stack,options)
            print('Q_k',Metric['Q_k'])
            if np.max(np.abs(Metric['Q_k']))<0.00001:
                break
        #compute new value of smooth part of objective
        f_kp1 = fun_f(x_kp1,model,options);
        #compute new objective value
        h_kp1 = f_kp1 + fun_g(x_kp1,model,options);
        #check breaking condition
        res = residual(x_kp1, res0,model,options);
        if res < tol:
            breakvalue = 2;
        
        #print info
        if (iter%check ==0):
            
            print('iter:%d, time:%5f, tau:%f,res:%f'%(iter,stime,tau,res))
        #handle breaking condition
        if breakvalue == 2:
            print('Tolerence calue reached');
        # tape residual
        time = time + (clock.time() - stime);
        if options['storeResidual'] == True:
            seq_res[iter] = res;
        if options['storeTime'] == True:
            seq_time[iter] = time;
        if options['storePoints'] == True:
            seq_x[:,iter] = x_kp1;
        if options['storeObjective'] == True:
            seq_obj[iter] = h_kp1;
        #if options['storeBeta'] == True:
        #    seq_beta[iter-1] = beta;
        
#return results
    output={
            'sol': x_kp1,
            'seq_obj': seq_obj,
            'seq_time':seq_time,
            'breakvalue':breakvalue
            }
    if options['storeResidual'] == True:
        output['seq_res'] = seq_res;
        
    return output
    
def cal_t(t_k,q_k,q_kp1):
    temp = 1 -q_k*t_k**2
    
    t_kp1 =0.5*(temp+ np.sqrt(temp**2+4*t_k**2*q_k/q_kp1))
    if t_kp1 <0:
        print('WARNING: t is negative')
    return t_kp1


def cal_beta(t_k,t_kp1,tau_kp10,mu_g,mu,mu_f):
    temp = 1+tau_kp10*mu_g - t_kp1 *tau_kp10*mu
    de   = 1-tau_kp10*mu_f
    beta = ((t_k-1)/t_kp1)*temp/de
    return beta


def checkCB(x_kp1,x_k,f_kp1,f_k,grad_k):
    
    
    return f_kp1 - f_k - grad_k.dot(x_kp1 -x_k)
































