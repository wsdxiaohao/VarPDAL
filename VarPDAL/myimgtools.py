import sys

from mymath import *
import matplotlib.image as mpimg
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
import scipy.sparse as sp
import scipy.sparse.linalg


# define a matrix that represents the application of filter to an 
# image of size M x N with M rows and N colums using reflecting 
# boundary conditions. 
def make_filter2D(M, N, filter):

    s = np.shape(filter)[0];   # filter size: s x s
    k = int((s-1)/2);       # filter center (k,k)
    
    row = np.zeros(s*s*M*N);
    col = np.zeros(s*s*M*N); 
    val = np.zeros(s*s*M*N); 
    ctr = 0;
    for y in range(0,M):
        for x in range(0,N):

            mat_row_idx = x*M + y;

            for j in range(0,s):
                for i in range(0,s):
                    ii = x + (i-k);
                    jj = y + (j-k);
                    if (ii<0):
                        ii = -ii-1;
                    if (jj<0):
                        jj = -jj-1;
                    if (ii >= N):
                        ii = 2*N - 1 - ii;
                    if (jj >= M):
                        jj = 2*M - 1 - jj;

                    #if x == N-2 and y == 5:
                    #    print ("pixel: (%d, %d), filter: (%d, %d), filter-pixel: (%d,%d)" % (x,y,i,j,ii,jj));

                    mat_col_idx = ii*M + jj;
                    row[ctr] = mat_row_idx;
                    col[ctr] = mat_col_idx;
                    val[ctr] = filter[j,i];
                    ctr = ctr + 1;

    A = csr_matrix((val, (row, col)), shape=(M*N, M*N));

    return A;




def make_derivatives2D(M, N):
    
    # y-derivatives
    row = zeros(2*M*N);
    col = zeros(2*M*N); 
    val = zeros(2*M*N); 
    ctr = 0;
    for x in range(0,N):
        for y in range(0,M-1):
            row[ctr] = x*M + y;
            col[ctr] = x*M + y;
            val[ctr] = -1.0;
            ctr = ctr + 1;
            
            row[ctr] = x*M + y;
            col[ctr] = x*M + y+1;
            val[ctr] = 1.0;
            ctr = ctr + 1;
    
    Ky = csr_matrix((val, (row, col)), shape=(M*N, M*N));

    # x-derivatives
    row = zeros(2*M*N);
    col = zeros(2*M*N); 
    val = zeros(2*M*N); 
    ctr = 0;
    for y in range(0,M):
        for x in range(0,N-1):
            row[ctr] = x*M + y;
            col[ctr] = x*M + y;
            val[ctr] = -1.0;
            ctr = ctr + 1;
            
            row[ctr] = x*M + y;
            col[ctr] = (x+1)*M + y;
            val[ctr] = 1.0;
            ctr = ctr + 1;
   

    Kx = csr_matrix((val, (row, col)), shape=(M*N, M*N));

    # x- and y-derivative (discrete gradient)
    K = sp.vstack([Kx,Ky]);

    return K;


# convert rgb image to gray scale image
def rgb2gray(img):
    if (img.shape[2]>1):
        return np.dot(img[...,:3], [0.299, 0.587, 0.114])
    else:
        return img;



