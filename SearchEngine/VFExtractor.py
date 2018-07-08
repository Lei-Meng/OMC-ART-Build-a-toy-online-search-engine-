# -*- coding: utf-8 -*-

import math
import numpy as np
from PIL import Image


def VFExtractor(direc,normalization):
    '''
    % direc: directory of a JPG image

    % normalization: a vector of size 2*225, storing the minimum and maximum
    % values for each of the visual features

    % cmVec: a color moment feature vector of the input image. The feature
    % vector is extracted from a 5*5 grid and represented by the first 3
    % moments for each grid region in Lab color space as a normalized
    % 225-dimensional vector.
     '''

    #%Step1: read image as a m*n*3 matrix
    jpgfile = Image.open(direc)
    jpgfile = np.array(jpgfile)


    #%Step 2: transform original image in RGB color space to Lab color space

    R = jpgfile[:,:,0]/255
    G = jpgfile[:,:,1]/255
    B = jpgfile[:,:,2]/255

    M,N=R.shape
    s = M*N

    #% Set a threshold
    T = 0.008856

    a=np.reshape(R,(1,s),order='F')
    b=np.reshape(G,(1,s),order='F')
    c=np.reshape(B,(1,s),order='F')

    RGB=np.concatenate([a,b,c])

    #% RGB to XYZ
    MAT = np.array([[0.412453,0.357580,0.180423],[0.212671,0.715160,0.072169],[0.019334,0.119193,0.950227]])

    XYZ = np.dot(MAT,RGB)

    X = XYZ[0,:] / 0.950456
    Y = XYZ[1,:]
    Z = XYZ[2,:] / 1.088754

    XT=np.ones((s))
    YT=np.ones((s))
    ZT=np.ones((s))


    for i in range(0, s):
        if X[i] > T:
            XT[i]=1
        else:
            XT[i]=0

        if Y[i] > T:
            YT[i]=1
        else:
            YT[i]=0

        if Z[i] > T:
            ZT[i]=1
        else:
            ZT[i]=0

    fX = XT * (X**(1/3)) + (1-XT) * (7.787 * X + 16/116)

    #% Compute L

    Y3 = Y**(1/3)
    fY = YT * Y3 + (1-YT) * (7.787 * Y + 16/116)
    L  = YT * (116 * Y3 - 16.0) + (1-YT) * (903.3 * Y)

    fZ = ZT * (Z**(1/3)) + (1-ZT) * (7.787 * Z + 16/116)


    #% Compute a and b
    a = 500 * (fX - fY)
    b = 200 * (fY - fZ)

    L = np.reshape(L, (M, N),order='F')
    a = np.reshape(a, (M, N),order='F')
    b = np.reshape(b, (M, N),order='F')

    labfile = np.array([L,a,b]) #%the image file in Lab color space


    #%Step 3: calculate the feature vector
    a,b,c = labfile.shape
    #% m, n represent sizes of the grid
    m = math.floor(b/5)
    n = math.floor(c/5)
    cmVec = np.zeros((225))

    for i in range(1, 6):
        for j in range(1, 6):
            subimage = labfile[:,(i-1)*m:i*m,(j-1)*n:j*n] #a trick is a[1:3] means [a[1],a[2]], no a[3]
            #% calculate the Mean values ...
            tmp = (i-1)*5+j-1
            cmVec[tmp*9] = np.mean(subimage[0,:,:])
            cmVec[tmp*9+1] = np.mean(subimage[1,:,:])
            cmVec[tmp*9+2] = np.mean(subimage[2,:,:])
            #% cal the second and third Moment values
            for p in range(1, m+1):
                for q in range(1, n+1):
                    cmVec[tmp*9+3] = cmVec[tmp*9+3] + (subimage[0,p-1,q-1]-cmVec[tmp*9])**2
                    cmVec[tmp*9+4] = cmVec[tmp*9+4] + (subimage[1,p-1,q-1]-cmVec[tmp*9+1])**2
                    cmVec[tmp*9+5] = cmVec[tmp*9+5] + (subimage[2,p-1,q-1]-cmVec[tmp*9+2])**2
                    #% ===
                    cmVec[tmp*9+6] = cmVec[tmp*9+6] + (subimage[0,p-1,q-1]-cmVec[tmp*9])**3
                    cmVec[tmp*9+7] = cmVec[tmp*9+7] + (subimage[1,p-1,q-1]-cmVec[tmp*9+1])**3
                    cmVec[tmp*9+8] = cmVec[tmp*9+8] + (subimage[2,p-1,q-1]-cmVec[tmp*9+2])**3


            cmVec[(tmp*9+3):(tmp*9+9)] = cmVec[(tmp*9+3):(tmp*9+9)]/(m*n)
            cmVec[tmp*9+3] = cmVec[tmp*9+3]**(1/2)
            cmVec[tmp*9+4] = cmVec[tmp*9+4]**(1/2)
            cmVec[tmp*9+5] = cmVec[tmp*9+5]**(1/2)

            if cmVec[tmp*9+6] > 0:
                 cmVec[tmp*9+6] = cmVec[tmp*9+6]**(1/3)
            else:
                cmVec[tmp*9+6] = -((-cmVec[tmp*9+6])**(1/3))

            if cmVec[tmp*9+7] > 0:
                 cmVec[tmp*9+7] = cmVec[tmp*9+7]**(1/3)
            else:
                cmVec[tmp*9+7] = -((-cmVec[tmp*9+7])**(1/3))

            if cmVec[tmp*9+8] > 0:
                 cmVec[tmp*9+8] = cmVec[tmp*9+8]**(1/3)
            else:
                cmVec[tmp*9+8] = -((-cmVec[tmp*9+8])**(1/3))


    a1 = cmVec ** 2
    a2 = sum(a1)
    temp = a2 ** (1 / 2)

    if temp != 0:
        cmVec = cmVec / temp

    # % Normalize the values to be in range of [0,1] using min-max normalization
    for i in range(0, 225):
        if cmVec[i] <= normalization[0,i]:
            cmVec[i]=0
        elif cmVec[i]>= normalization[1,i]:
            cmVec[i]=1
        else:
            cmVec[i]=(cmVec[i]-normalization[0,i])/(normalization[1,i]-normalization[0,i])

    return cmVec













