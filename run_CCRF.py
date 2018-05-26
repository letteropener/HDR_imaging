import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys

CCRF_LUT_R = np.zeros((256,256))
CCRF_LUT_G = np.zeros((256,256))
CCRF_LUT_B = np.zeros((256,256))

def run_CCRF(f2q,fq):
    CCRF = np.zeros(fq.shape)
    for i in range(f2q.shape[0]):
        for j in range(f2q.shape[1]):
            X_B = fq[i][j][0]
            Y_B = f2q[i][j][0]
            CCRF[i][j][0] = CCRF_LUT_B[X_B][Y_B]
            X_G = fq[i][j][1]
            Y_G = f2q[i][j][1]
            CCRF[i][j][1] = CCRF_LUT_G[X_G][Y_G]
            X_R = fq[i][j][2]
            Y_R = f2q[i][j][2]
            CCRF[i][j][2] = CCRF_LUT_R[X_R][Y_R]
    return CCRF

CCRF_LUT_R = np.loadtxt('img/CCRF_R.txt')
CCRF_LUT_G = np.loadtxt('img/CCRF_G.txt')
CCRF_LUT_B = np.loadtxt('img/CCRF_B.txt')

f2q = cv2.imread("img/ccrf_6.jpg")
fq = cv2.imread("img/ccrf_5.jpg")
ccrf = run_CCRF(f2q,fq)
cv2.imwrite('img/ccrf_1_5.jpg',ccrf)