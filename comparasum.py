import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(x, a, c):
    # k = 2
    return x * np.power(2, a * c) / np.power((1 + np.power(x, 1 / c) * (np.power(2, a) - 1)), c)


def matadd(m1,m2):
    result = np.zeros(m1.shape)
    assert m1.shape == m2.shape
    for i in range(m1.shape[0]):
        for j in range(m1.shape[1]):
            result[i][j] = m1[i][j] + m2[i][j]
    return result

def bin_count(comparagram,color):
    fit_curve = np.zeros(256)
    for i in range(256):
        indices = np.argmax(comparagram[:,i])
        #print(indices)
        fit_curve[i] = indices.mean()
    np.savetxt("result/fit_" + color + "fit.txt", fit_curve, fmt="%d")
    x_axis = np.array(range(256))
    plt.plot(x_axis[fit_curve > 0],fit_curve[fit_curve > 0], c='black', linewidth=2.0)
    plt.savefig("result/fit_"+ color + ".jpg")
    plt.clf()
    return fit_curve, x_axis

comp1_B = cv2.imread("result/out_B_1k_2k.jpg",cv2.IMREAD_GRAYSCALE)
comp2_B = cv2.imread("result/out_B_2k_4k.jpg",cv2.IMREAD_GRAYSCALE)
comp3_B = cv2.imread("result/out_B_4k_8k.jpg",cv2.IMREAD_GRAYSCALE)
comp4_B = cv2.imread("result/out_B_8k_16k.jpg",cv2.IMREAD_GRAYSCALE)
comp5_B = cv2.imread("result/out_B_16k_32k.jpg",cv2.IMREAD_GRAYSCALE)
comp6_B = cv2.imread("result/out_B_32k_64k.jpg",cv2.IMREAD_GRAYSCALE)

compsum_B = matadd(matadd(matadd(matadd(matadd(comp1_B.astype(float),comp2_B.astype(float)),comp3_B.astype(float)),comp4_B.astype(float)) \
    ,comp5_B.astype(float)),comp6_B.astype(float))

#np.savetxt("result/test.txt",compsum_B,fmt="%d")
compsum_B[compsum_B>255] = 255
#print(compsum_B.max())
np.savetxt("result/compsum_B.txt",compsum_B,fmt="%d")
sum_B = Image.fromarray(np.uint8(compsum_B,mode='L'))
sum_B.save('result/compsum_B.jpg')

fit_B,x_B = bin_count(np.flip(compsum_B,0),'B')
x_data = (x_B+0.5) / 256.
y_data = (fit_B+0.5) / 256.
plt.plot(x_data,y_data)
popt, pcov = curve_fit(func, x_data, y_data)
plt.plot(x_data, func(x_data, *popt), 'r-',label='fit: a=%5.3f, c=%5.3f' % tuple(popt))
plt.legend()
plt.savefig('result/fit_curve_B.png')
plt.clf()




comp1_R = cv2.imread("result/out_R_1k_2k.jpg",cv2.IMREAD_GRAYSCALE)
comp2_R = cv2.imread("result/out_R_2k_4k.jpg",cv2.IMREAD_GRAYSCALE)
comp3_R = cv2.imread("result/out_R_4k_8k.jpg",cv2.IMREAD_GRAYSCALE)
comp4_R = cv2.imread("result/out_R_8k_16k.jpg",cv2.IMREAD_GRAYSCALE)
comp5_R = cv2.imread("result/out_R_16k_32k.jpg",cv2.IMREAD_GRAYSCALE)
comp6_R = cv2.imread("result/out_R_32k_64k.jpg",cv2.IMREAD_GRAYSCALE)


compsum_R = matadd(matadd(matadd(matadd(matadd(comp1_R.astype(float),comp2_R.astype(float)),comp3_R.astype(float)),comp4_R.astype(float)) \
    , comp5_R.astype(float)),comp6_R.astype(float))

compsum_R[compsum_R>255] = 255
#print(compsum_B.max())
np.savetxt("result/compsum_R.txt",compsum_R,fmt="%d")
sum_R = Image.fromarray(np.uint8(compsum_R,mode='L'))
sum_R.save('result/compsum_R.jpg')



fit_R,x_R = bin_count(np.flip(compsum_R,0),'R')
x_data = (x_R+0.5) / 256.
y_data = (fit_R+0.5) / 256.
plt.plot(x_data,y_data)
popt, pcov = curve_fit(func, x_data, y_data)
plt.plot(x_data, func(x_data, *popt), 'r-',label='fit: a=%5.3f, c=%5.3f' % tuple(popt))
plt.legend()
plt.savefig('result/fit_curve_R.png')
plt.clf()


comp1_G = cv2.imread("result/out_G_1k_2k.jpg",cv2.IMREAD_GRAYSCALE)
comp2_G = cv2.imread("result/out_G_2k_4k.jpg",cv2.IMREAD_GRAYSCALE)
comp3_G = cv2.imread("result/out_G_4k_8k.jpg",cv2.IMREAD_GRAYSCALE)
comp4_G = cv2.imread("result/out_G_8k_16k.jpg",cv2.IMREAD_GRAYSCALE)
comp5_G = cv2.imread("result/out_G_16k_32k.jpg",cv2.IMREAD_GRAYSCALE)
comp6_G = cv2.imread("result/out_G_32k_64k.jpg",cv2.IMREAD_GRAYSCALE)

compsum_G = matadd(matadd(matadd(matadd(matadd(comp1_G.astype(float),comp2_G.astype(float)),comp3_G.astype(float)),comp4_G.astype(float)) \
    , comp5_G.astype(float)), comp6_G.astype(float))

compsum_G[compsum_G>255] = 255
#print(compsum_B.max())
np.savetxt("result/compsum_G.txt",compsum_G,fmt="%d")
sum_G = Image.fromarray(np.uint8(compsum_G,mode='L'))
sum_G.save('result/compsum_G.jpg')

fit_G,x_G = bin_count(np.flip(compsum_G,0),'G')
x_data = (x_G+0.5) / 256.
y_data = (fit_G+0.5) / 256.
plt.plot(x_data,y_data)
popt, pcov = curve_fit(func, x_data, y_data)
plt.plot(x_data, func(x_data, *popt), 'r-',label='fit: a=%5.3f, c=%5.3f' % tuple(popt))
plt.legend()
plt.savefig('result/fit_curve_G.png')
plt.clf()