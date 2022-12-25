import cv2
from PIL import Image
import numpy as np
import math
import cmath
import random
from scipy import stats
import scipy
from scipy.special import erfinv
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans as KNN


def decompose(img):
    r=img[:,:,0]
    g=img[:,:,1]
    b=img[:,:,2]
    return r,g,b

def fft(img,inverse=False,absolute=False):
    r=img[:,:,0]
    g=img[:,:,1]
    b=img[:,:,2]
    if(not inverse):
        r=np.fft.fft2(r)
        g=np.fft.fft2(g)
        b=np.fft.fft2(b)
    else:
        r=np.fft.ifft2(r)
        g=np.fft.ifft2(g)
        b=np.fft.ifft2(b)
    if(absolute):
        return np.abs(np.dstack((r,g,b)))
    else:
        return np.dstack((r,g,b))



def mapper(image,length,height,factor=1.0):
    map1=[]
    map2=[]
    i=-1
    while(i<1):
        lne1=[]
        lne2=[]
        j=-1
        while(j<1):
            x=j
            y=i
            #Here, x and y live in the interval [-1,1], normalized by the size of the input image
            #Mapper function should also map input image to the [-1,1] interval, as the following
            #code takes care of the scaling
            x=x
            y=y

            x=(x+1)/2
            y=(y+1)/2
            x*=len(image[0])
            y*=len(image)
            lne1.append(x)
            lne2.append(y)
            j+=2*factor/height
        map1.append(lne1)
        map2.append(lne2)
        i+=2*factor/length
    map1=np.array(map1)
    map2=np.array(map2)
    map1=cv2.resize(map1,dsize=(length,height))
    map2=cv2.resize(map2,dsize=(length,height))
    map1, map2 = cv2.convertMaps(map1.astype(np.float32),map2.astype(np.float32), cv2.CV_16SC2)
    return cv2.remap(src=image,map1=map1,map2=map2,interpolation=cv2.INTER_AREA,borderMode=cv2.BORDER_REPLICATE)

image=cv2.imread('test1.png')
out=mapper(image,1000,500,factor=1)
cv2.imwrite('out.png',out)