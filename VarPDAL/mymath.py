import numpy as np
from numpy.random import rand
from numpy.random import normal as randn
from numpy import abs,sum,max,sign,sqrt
from numpy import zeros,ones

import matplotlib.pyplot as plt






################################################################################
#computing total variation norm of kx
def TVnorm(n,Kx):
    N=2*n #len of Kx
    
    Dx_2=(Kx*(Kx));# entrywise squared Kx
    Dx_tv=np.sum(np.sqrt(Dx_2[0:n]+Dx_2[n:N])); #Total variation of Dx
    return Dx_tv








################################################################################
### Diagonally Scaled Proximal Opertators ######################################
################################################################################
#                                                                              #
# These proximal operators are functions prox_{g}^D that assign to a point x0  #
# a solution of the following optimization problem:                            #
#                                                                              #
#       min_x g(x) + 0.5|x-x0|_{D}^2                                           #
#                                                                              #
################################################################################

def prox_zero(x0, d, params={}):
    """
    Proximal mapping for the zero function = identity mapping.
    """
    return x0;


def prox_sql2(x0, d, params={}):
    """
    Proximal mapping for the function

        g(x) = 0.5*|x|_2^2

    """
    return x0/(1.0 + 1.0/d);


def prox_l1(x0, d, params={}):
    """ 
    Proximal mapping for the function

        g(x) = |x|_1

    The solution is the soft-shrinkage thresholding.
    """
    return np.maximum(0.0, abs(x0) - 1.0/d)*sign(x0);

def prox_groupl2l1(x0, d, params={}):
    """ 
    Proximal mapping for the function

        g(x) = |x|_B 
    
    where
        
        B       [0,K_1,K_2,...,N] is a list of coordinates belonging to the 
                same group. It contains len(B)-1 groups. The i-the group 
                (i=0,1,...,len(B)-1) contains the indizes {B[i], ..., B[i+1]-1}.
        |x|_B   := sum_{i=0}^{len(B)-1} |x_{B[i], ..., B[i+1]-1}|_2
        d       WARNING: The implementation requires that the coordinates
                of d belonging to the same group are equal!

    The solution is the group soft-shrinkage thresholding.
    """
    x = x0.copy();
    x_sq = (d*x)**2;
    B = params['B'];
    for k in range(0,len(B)-1):
        dnrm = sqrt(sum( x_sq[B[k]:B[k+1]] ));
        if (dnrm <= 1.0):
            x[B[k]:B[k+1]] = 0.0;
        else:
            x[B[k]:B[k+1]] = x[B[k]:B[k+1]] - x[B[k]:B[k+1]]/dnrm;
    return x;


def prox_l0(x0, d, params={}):
    """ 
    Proximal mapping for the function

        g(x) = |x|_0 = |{x_i != 0}|

    The solution is a hard shrinkage.
    """
    x = x0.copy();
    idx = (x*x)<=2.0/d;
    x[idx] = 0;
    return x;


################################################################################
### Projection Operators #######################################################
################################################################################
#                                                                              #
# Projection opertors are proximal operators for indicator functions of sets.  #
# It computes a closest point from x0 in a set C. In other words, the output   #
# solves the following:                                                        #
#                                                                              #
#       min_x \ind_C(x) + 0.5|x-x0|^2                                          #
#                                                                              #
# where                                                                        #
#                                                                              #
#       \ind_C  is the indicator function of the set C                         #
# 	x0	proximal center (point in R^N)                                 #
#                                                                              #
################################################################################

def proj_simplex(x0, d=1.0, params={}):
    """ 
    Projects the point x0 onto the unit simplex.
    """
    N = x0.size;

    mu = -1e10;
    for i in range(0,N):
        a = 0;
        b = 0;
        for j in range(0,N):
            if (x0[j,0] > mu):
                a = a + x0[j,0];
                b = b + 1;
        mu = (a-1)/b;
   
    x0 = x0 - mu;
    x0[x0<=0] = 0;
    return x0;


def proj_box(x0, d=1.0, params={'a':0.0, 'b':1.0}):
    """ 
    Projects the point x0 onto a box of size [a,b]^N.
    """
    a = params['a'];
    b = params['b'];
    
    return pmax(a, pmin(b, x0));


################################################################################
### Proximal Calculus ##########################################################


def prox_calc_add_lin(x0, d, prox, a, params={}):
    """
    The function g in the proximal mapping is modified by a linear term, i.e.,
    the proximal mapping is computed with respect to

        g(x) + <x,a>.

    where

        a       is a vector in R^N.

    """
    return prox(x0 - d*a, d, params);

def prox_calc_shift_lin(x0, d, prox, s, a, params={}):
    """
    The function g in the proximal mapping is shifted by a linear transform, 
    i.e., the proximal mapping is computed with respect to

        g(s*x - a)

    where

        s       is a scalar
        a       is a vector in R^N.

    """
    return (prox(s*x0 - a, d/(s**2)) + a, params)/s;


