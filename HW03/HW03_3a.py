####################################################
## Ryan Gentry                                    ##
## ECE 6254 - Statistical Machine Learning        ##
## HW 3                                           ##
## Problem 3a: Training an SVM wiht inhomogenous  ##
## linear and quadratic kernels                   ##
## 18 February 2017                               ##
####################################################

# Set-up libraries and import mldata 
from sklearn.datasets import fetch_mldata
import numpy as np
from sklearn import svm
import math
import matplotlib.pyplot as plt
mnist = fetch_mldata('MNIST original')

# Problem 3: put data into more convenient form and plot test image
X = mnist.data
y = mnist.target
plt.title('The 3rd image is a {label}'.format(label=int(y[2])))
plt.imshow(X[2].reshape((28,28)),cmap='gray')
plt.show()
print("X is %d x %d and Y is %d x %x" % (len(X),len(X[0]),len(y),1))

# Build training & testing sets to classify between the images of the digits 4 & 9
# Separate training set into fit & holdout sets to choose best SVM parameter C
import numpy as np
import random
X4 = X[y==4,:]
y4 = y[y==4]
X9 = X[y==9,:]
y9 = y[y==9]
Xfit = np.concatenate((X4[0:2000],X9[0:2000]),axis=0)
yfit = np.concatenate((y4[0:2000],y9[0:2000]),axis=0)
Xholdout = np.concatenate((X4[2000:4000],X9[2000:4000]),axis=0)
yholdout = np.concatenate((y4[2000:4000],y9[2000:4000]),axis=0)
Xtrain = np.concatenate((X4[0:4000],X9[0:4000]),axis=0)
ytrain = np.concatenate((y4[0:4000],y9[0:4000]),axis=0)
Xtest = np.concatenate((X4[4000:],X9[4000:]),axis=0)
ytest = np.concatenate((y4[4000:],y9[4000:]),axis=0)

# Use built in solver to find best C parameter for poly 1st degree SVM on training dataset
# First choose range of C's to test, train each C on Xfit/yfit
#def polysolv(Pe_min,R,s,deg,n):
#    X = np.linspace(R-s/2,R+s/2,n)
#    C = 10**X
#    Pe = np.zeros(len(C))
#    for i in range(0,len(C)):
#        clf = svm.SVC(C[i],kernel='poly',degree=deg)
#        clf.fit(Xfit,yfit)
#        Pe[i]=1-clf.score(Xholdout,yholdout)
#        print("Deg = %d, Spread = %.1f, C = %.6f, Pe = %.6f" % (deg, s, C[i],Pe[i]))
#    Pe_min = np.amin(Pe)
#    Cmin = C[np.argmin(Pe)]
#    R = math.log(Cmin,10)
#    print("Round finished, Pe_min = %.6f, Cmin = %.4f & log(Cmin) = %.4f" % (Pe_min, Cmin, R))
#    return R,Cmin, Pe_min

def rbfsolv(Cc,gc,n,s):
    Xc = np.linspace(Cc-s,Cc+s,n)
    C = 10**Xc
    Xg = np.linspace(gc-s,gc+s,n)
    gamma = 10**Xg
    Pe = np.zeros((len(C),len(gamma)))
    for i in range(0,len(C)):
        for j in range(0,len(gamma)):
            clf = svm.SVC(C[i],kernel='rbf',gamma=gamma[j])
            clf.fit(Xfit,yfit)
            Pe[i][j] = 1-clf.score(Xholdout,yholdout)
            print("C=%.2f and gamma = %.8f give Pe = %.4f" % (C[i],gamma[j],Pe[i][j]))
    minval = np.amin(Pe)
    argmin = np.argmin(Pe)
    if(argmin<n): R_Cc=0
    else: R_Cc = argmin/n
    Cc = C[R_Cc]
    R_gc = argmin%n
    print("R_gc = %d, R_Cc = %d, argmin = %d, Cc = %d, n = %d, minval = %.6f" % (R_gc, R_Cc, argmin, Cc, n, minval))
    gc = gamma[R_gc-1]
    print("Returning center C value = %.1f, center g value = %.8f, and min Pe value = %.6f" % (Cc, gc, minval))
    return Cc, gc, minval

# Inhomogeneous linear kernel first, then quadratic kernel
tol = 1e-5
#Pe_last = 1
#deg = [1,2]
#Pe_fin = np.zeros(3)
#for d in deg:
#    R = 0
#    s = 10
#    n = 11
#    delta = Pe_last
#    while (delta > tol):
#        R, Cmin, Pe_min = polysolv(Pe_last,R,s,d,n)
#        delta = Pe_last - Pe_min
#        Pe_last = Pe_min
#        s = s/5
#    clf = svm.SVC(Cmin,kernel = 'poly',degree = d)
#    clf.fit(Xtrain,ytrain)
#    Pe_fin = 1-clf.score(Xtest,ytest)
#    if(d==1): 
#        print("For inhomogeneous linear kernel:")
#        sv_lin = clf.support_vectors_
#    else: 
#        print("For quadratic kernel:")
#        sv_quad = clf.support_vectors_
#    print("Best C = %.6f, where probability of error = %.4f and support vectors = %s" % (Cmin, Pe_fin, clf.support_vectors_.shape))

# Problem 3b: Now use radial basis function kernel, and determine best gamma
Pe_last = 1
R_Cc = 5 # Rank of best guess for center of starting C values (10^5)
R_gc = -5 # Rank of best guess for center of starting gamma values (10^-5)
n = 5 # Resolution of tests in rbfsolv = number of values for both C and gamma tested per iteration
s = 5 # Starting spread of rank for C (10^0 - 10^10) and gamma (10^-10 - 10^0) values 
delta = Pe_last
while(delta > tol):
    Cc,gc,minval = rbfsolv(R_Cc,R_gc,n,s)
    R_Cc = math.log(Cc,10)
    R_gc = math.log(gc,10)
    delta = Pe_last - minval
    Pe_last = minval
    s = s/4
clf = svm.SVC(Cc, kernel = 'rbf', gamma=gc)
clf.fit(Xtrain,ytrain)
Pe_fin = 1-clf.score(Xtest,ytest)
print("Best (C,gamma) = (%.6f,%.8f) where probability of error = %.4f and support vectors = %s" % (Cc, gc, Pe_fin, clf.support_vectors_.shape))

# Problem 3c: For each kernel, show images of 16 support vectors that violate margin by greatest amount

# for j in sv?:
#    f, axarr = plt.subplots(4,4)
#    axarr[0,0].imshow(X[j].reshape((28,28)),cmap='gray')
#    axarr[0,0].set_title('{label}'.format(label=int(y[j])))