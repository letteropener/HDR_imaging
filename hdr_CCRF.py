import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
EPSILON=sys.float_info.epsilon
# R channel
R_a = 29.170
R_c = 0.031
# G channel
G_a = 30.541
G_c = 0.030
# B channel
B_a = 23.232
B_c = 0.037

f_inverse_LUT_R = {}
f_inverse_LUT_G = {}
f_inverse_LUT_B = {}

f_LUT_R = {}
f_LUT_G = {}
f_LUT_B = {}

certainty_LUT_R = {}
certainty_LUT_G = {}
certainty_LUT_B = {}

HDR_LUT_R = np.zeros((256,256))
HDR_LUT_G = np.zeros((256,256))
HDR_LUT_B = np.zeros((256,256))

CCRF_LUT_R = np.zeros((256,256))
CCRF_LUT_G = np.zeros((256,256))
CCRF_LUT_B = np.zeros((256,256))


def f(q,a,c,func=None):
    if func is not None:
        return func(q)
    q_a=np.power(q,a)
    return np.power((q_a/(1+q_a)),c)

def f_inverse(f,a,c,func=None):
    if func is not None:
        return func(f)

    f_cth_root=np.power(f,1/c)
    return np.power( f_cth_root/(1-f_cth_root) ,1/a)

def conf(q,ki,a,c):
    q=np.divide(q,ki)
    q_a = np.power(q, a)
    return a*c * np.divide(np.power((1/(1+(1/q_a))),(1+c)),q_a)

def create_certainty_LUT():
    for i in range(256):
        qR_a = np.power(f_inverse_LUT_R[i], R_a)
        certainty_LUT_R[f_inverse_LUT_R[i]] = R_a*R_c * np.divide(np.power((1/(1+(1/qR_a))),(1+R_c)),qR_a)
        qG_a = np.power(f_inverse_LUT_G[i], G_a)
        certainty_LUT_G[f_inverse_LUT_G[i]] = G_a*G_c * np.divide(np.power((1/(1+(1/qG_a))),(1+G_c)),qG_a)
        qB_a = np.power(f_inverse_LUT_B[i], B_a)
        certainty_LUT_B[f_inverse_LUT_B[i]] = B_a*B_c * np.divide(np.power((1/(1+(1/qB_a))),(1+B_c)),qB_a)

def create_f_inverse_LUT():
    for f in range(256):
        f_inverse_LUT_R[f] = f_inverse((f + 0.5) / 256., a=R_a, c=R_c)
        f_inverse_LUT_G[f] = f_inverse((f + 0.5) / 256., a=G_a, c=G_c)
        f_inverse_LUT_B[f] = f_inverse((f + 0.5) / 256., a=B_a, c=B_c)

def create_f_LUT():
    for f in range(256):
        f_LUT_R[f_inverse_LUT_R[f]] = f
        f_LUT_G[f_inverse_LUT_G[f]] = f
        f_LUT_B[f_inverse_LUT_B[f]] = f

def round_matrix(m):
    ret = np.zeros((256,256))
    for i in range(256):
        for j in range(256):
            ret[i][j] = int(round(m[i][j]))
    return ret


def create_HDR_LUT_R():
    total_conf = np.zeros((256,256))
    ret = np.zeros((256,256))
    for f2q in range(256):
        for fq in range(256):
            lightspace_q = f_inverse((fq+0.5)/256.,a=R_a, c=R_c)
            lightspace_2q = f_inverse((f2q+0.5)/256.,a=R_a, c=R_c)
            cur_conf = conf(lightspace_q, ki=1,a=R_a, c=R_c) / 1
            total_conf[fq,f2q] += conf(lightspace_q, ki=1,a=R_a, c=R_c)
            ret[fq,f2q] += np.multiply(cur_conf, lightspace_q)
            cur_conf = conf(lightspace_2q, ki=2,a=R_a, c=R_c) / 2
            total_conf[fq, f2q] += conf(lightspace_2q, ki=2,a=R_a, c=R_c)
            ret[fq, f2q] += np.multiply(cur_conf, lightspace_2q)

    ret = np.divide(ret, total_conf)
    HDR_LUT_R =  f(ret,a=R_a, c=R_c)*256.-0.5
    HDR_LUT_R = round_matrix(HDR_LUT_R)
    #print(HDR_LUT_R)
    return HDR_LUT_R



def create_HDR_LUT_G():
    total_conf = np.zeros((256,256))
    ret = np.zeros((256,256))
    for f2q in range(256):
        for fq in range(256):
            lightspace_q = f_inverse((fq+0.5)/256.,a=G_a, c=G_c)
            lightspace_2q = f_inverse((f2q+0.5)/256.,a=G_a, c=G_c)
            cur_conf = conf(lightspace_q, ki=1,a=G_a, c=G_c) / 1
            total_conf[fq,f2q] += conf(lightspace_q, ki=1,a=G_a, c=G_c)
            ret[fq,f2q] += np.multiply(cur_conf, lightspace_q)
            cur_conf = conf(lightspace_2q, ki=2,a=G_a, c=G_c) / 2
            total_conf[fq, f2q] += conf(lightspace_2q, ki=2,a=G_a, c=G_c)
            ret[fq, f2q] += np.multiply(cur_conf, lightspace_2q)

    ret = np.divide(ret, total_conf)
    HDR_LUT_G =  f(ret,a=G_a, c=G_c)*256.-0.5
    HDR_LUT_G = round_matrix(HDR_LUT_G)
    #print(HDR_LUT_G)
    return HDR_LUT_G

def create_HDR_LUT_B():
    total_conf = np.zeros((256,256))
    ret = np.zeros((256,256))
    for f2q in range(256):
        for fq in range(256):
            lightspace_q = f_inverse((fq+0.5)/256.,a=B_a, c=B_c)
            lightspace_2q = f_inverse((f2q+0.5)/256.,a=B_a, c=B_c)
            cur_conf = conf(lightspace_q, ki=1,a=B_a, c=B_c) / 1
            total_conf[fq,f2q] += conf(lightspace_q, ki=1,a=B_a, c=B_c)
            ret[fq,f2q] += np.multiply(cur_conf, lightspace_q)
            cur_conf = conf(lightspace_2q, ki=2,a=B_a, c=B_c) / 2
            total_conf[fq, f2q] += conf(lightspace_2q, ki=2,a=B_a, c=B_c)
            ret[fq, f2q] += np.multiply(cur_conf, lightspace_2q)

    ret = np.divide(ret, total_conf)
    HDR_LUT_B =  f(ret,a=B_a, c=B_c)*256.-0.5
    HDR_LUT_B = round_matrix(HDR_LUT_B)
    #print(HDR_LUT_B)
    return HDR_LUT_B


def run_HDR(f2q,fq):
    HDR = np.zeros(fq.shape)
    for i in range(f2q.shape[0]):
        for j in range(f2q.shape[1]):
            X_B = fq[i][j][0]
            Y_B = f2q[i][j][0]
            HDR[i][j][0] = HDR_LUT_B[X_B][Y_B]
            X_G = fq[i][j][1]
            Y_G = f2q[i][j][1]
            HDR[i][j][1] = HDR_LUT_G[X_G][Y_G]
            X_R = fq[i][j][2]
            Y_R = f2q[i][j][2]
            HDR[i][j][2] = HDR_LUT_R[X_R][Y_R]
    return HDR

def sigmaF1(comparasum):
    sigma = np.zeros((256,))
    sum_col = np.zeros((256,))
    cdf_col = np.zeros((256,256))
    sum_col_tmp = 0
    Q1 = np.zeros((256,))
    Q3 = np.zeros((256,))
    IQR = np.zeros((256,))
    # create cdf_col
    for i in range(256):
        for j in range(256):
            sum_col_tmp += comparasum[j][i]
            cdf_col[j][i] = sum_col_tmp
        sum_col_tmp = 0
    np.savetxt('img/cdf_col.txt',cdf_col,fmt="%d")
    for i in range(256):
        sum_col[i] = np.sum(comparasum[:,i])
        Q1[i] = (sum_col[i]+1)*0.25
        Q3[i] = (sum_col[i]+1)*0.75
        for j in range(256):
            if (Q1[i] < cdf_col[j][i]):
                Q1[i] = j + (Q1[i]-cdf_col[j-1][i])/(cdf_col[j][i] - cdf_col[j-1][i])
                break
        for j in range(256):
            if (Q3[i] < cdf_col[j][i]):
                Q3[i] = j + (Q3[i]-cdf_col[j-1][i])/(cdf_col[j][i] - cdf_col[j-1][i])
                break
    IQR = Q3 - Q1
    sigma = np.divide(IQR,1.349)
    return sigma

def sigmaF2(comparasum):
    # sigma of all rows
    sigma = np.zeros((256,))
    sum_row = np.zeros((256,))
    cdf_row = np.zeros((256, 256))
    sum_row_tmp = 0
    Q1 = np.zeros((256,))
    Q3 = np.zeros((256,))
    IQR = np.zeros((256,))
    # create cdf_col
    for i in range(256):
        for j in range(256):
            sum_row_tmp += comparasum[i][j]
            cdf_row[i][j] = sum_row_tmp
        sum_row_tmp = 0
    np.savetxt('img/cdf_row.txt', cdf_row, fmt="%d")
    for i in range(256):
        sum_row[i] = np.sum(comparasum[i,:])
        Q1[i] = (sum_row[i]+1)*0.25
        Q3[i] = (sum_row[i]+1)*0.75
        for j in range(256):
            if (Q1[i] < cdf_row[i][j]):
                Q1[i] = j + (Q1[i]-cdf_row[i][j-1])/(cdf_row[i][j] - cdf_row[i][j-1])
                break
        for j in range(256):
            if (Q3[i] < cdf_row[i][j]):
                Q3[i] = j + (Q3[i]-cdf_row[i][j-1])/(cdf_row[i][j] - cdf_row[i][j-1])
                break
    IQR = Q3 - Q1
    sigma = np.divide(IQR,1.349)
    return sigma



def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plotsigmas(sigmaf1,sigmaf2):
    print(sigmaf1)
    print(sigmaf2)
    plt.plot(sigmaf1,'ro',label='sigma(f1)-- unsmoothed')
    plt.plot(smooth(sigmaf1,2),'r',label='sigma(f1)-- smoothed')
    plt.plot(sigmaf2,'bo',label='sigma(f2)-- unsmoothed')
    plt.plot(smooth(sigmaf2,2),'b',label='sigma(f2)-- smoothed')
    plt.legend()
    plt.show()

def CCRF_argmin(f1,f2,sigmaf1,sigmaf2,a,c,f_inverse_LUT):
    results = np.zeros((256,))
    for i in range(256):
        q = f_inverse_LUT[i]
        q_2 = q*2
        results[i] = ((((f1+0.5)/256. - f(q,a, c))**2)/((sigmaf1[f1])**2)) + \
        ((((f2+0.5)/256. - f(q_2, a, c)) ** 2) / ((sigmaf2[f2]) ** 2))
    result = f_inverse_LUT[np.argmin(results)]
    return f(result,a,c) * 256. -0.5

def create_CCRF_LUT(f_inverse_LUT,sigmaf1,sigmaf2,color,a,c):
    CCRF = np.zeros((256,256))
    for i in range(256):
        for j in range(256):
            CCRF[i][j] = CCRF_argmin(i, j, sigmaf1, sigmaf2,a,c,f_inverse_LUT)
    CCRF = round_matrix(CCRF).astype(np.uint8)
    np.savetxt('img/CCRF_'+color+'.txt', CCRF, fmt="%d")
    output = Image.fromarray(CCRF, mode='L')
    output.save('img/CCRF_'+color+'.jpg')
    return CCRF


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

# CCRF
#compsum_B = np.loadtxt('result/compsum_B.txt')
#sigmaf1_B = sigmaF1(np.flip(compsum_B,0))
#sigmaf2_B = sigmaF2(np.flip(compsum_B,0))
#plotsigmas(sigmaf1,sigmaf2)
#compsum_G = np.loadtxt('result/compsum_G.txt')
#sigmaf1_G = sigmaF1(np.flip(compsum_G,0))
#sigmaf2_G = sigmaF2(np.flip(compsum_G,0))

#compsum_R = np.loadtxt('result/compsum_R.txt')
#sigmaf1_R = sigmaF1(np.flip(compsum_R,0))
#sigmaf2_R = sigmaF2(np.flip(compsum_R,0))

#create_f_inverse_LUT()
#CCRF_LUT_B = create_CCRF_LUT(f_inverse_LUT_B,sigmaf1_B,sigmaf2_B,'B',a = B_a, c = B_c)
#CCRF_LUT_G = create_CCRF_LUT(f_inverse_LUT_G,sigmaf1_G,sigmaf2_G,'G',a = G_a, c = G_c)
#CCRF_LUT_R = create_CCRF_LUT(f_inverse_LUT_R,sigmaf1_R,sigmaf2_R,'R',a = R_a, c = R_c)


#f2q = cv2.imread("img/32000.jpeg")
#fq = cv2.imread("img/64000.jpeg")
#ccrf = run_CCRF(f2q,fq)
#cv2.imwrite('img/ccrf.jpg',ccrf)





# Create LUTs
create_f_inverse_LUT()
create_f_LUT()
create_certainty_LUT()
HDR_LUT_R = create_HDR_LUT_R().astype(np.uint8)
HDR_LUT_G = create_HDR_LUT_G().astype(np.uint8)
HDR_LUT_B = create_HDR_LUT_B().astype(np.uint8)
np.savetxt('img/HDR_R.txt',HDR_LUT_R,fmt="%d")
Img_R = Image.fromarray(HDR_LUT_R,mode='L')
Img_R.save('img/R.jpg')
np.savetxt('img/HDR_G.txt',HDR_LUT_G,fmt="%d")
Img_G = Image.fromarray(HDR_LUT_G,mode='L')
Img_G.save('img/G.jpg')
np.savetxt('img/HDR_B.txt',HDR_LUT_B,fmt="%d")
Img_B = Image.fromarray(HDR_LUT_B,mode='L')
Img_B.save('img/B.jpg')

# import 2 images
f2q = cv2.imread("1.jpg")
fq = cv2.imread("2.jpg")
hdr = run_HDR(f2q,fq)
#cv2.imshow('1_s',fq)
#cv2.imshow('2_s',f2q)
#cv2.imshow('hdr',hdr/255)

cv2.imwrite('img/hdr.jpg',hdr)

#cv2.waitKey(0)
#cv2.destroyAllWindows()






