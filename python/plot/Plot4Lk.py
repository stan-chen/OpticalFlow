# -*- coding:utf-8 -*-
'''
    Author : Xuefei Chen
    Email : chenxuefei_pp@163.com
    Created on : 2017/3/6 9:17
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

def plot_3d(base_count):
    ZOOM_RATIO = 2
    shape = base_count.shape
    xCount = shape[0]
    yCount = shape[1]

    #x = np.arange(0,shape[0],1)
    #y = np.arange(0,shape[1],1)

    x = np.arange(0,xCount/ZOOM_RATIO,1)
    y = np.arange(0,yCount/ZOOM_RATIO,1)


    X,Y = np.meshgrid(x,y)

    Z = np.zeros(X.shape,np.float64)
    for idx in range(0,X.shape[0]):
        for idy in range(0,X.shape[1]):
            values = base_count[idx:(idx+1)*ZOOM_RATIO,idy:(idy+1)*ZOOM_RATIO]
            Z[idx][idy] = np.mean(values) #base_count[int(X[idx][idy])][int(Y[idx][idy])]

    #基于ax变量绘制三维图
    #xs表示x方向的变量
    #ys表示y方向的变量
    #zs表示z方向的变量，这三个方向上的变量都可以用list的形式表示
    #m表示点的形式，o是圆形的点，^是三角形（marker)
    #c表示颜色（color for short）
    #ax.scatter(xs, ys, zs, c = 'r', marker = '^') #点为红色三角形

    # C=[]
    # for z in Z:
    #     if z >= -60:
    #         C.append("r")
    #     elif z < -300:
    #         C.append("k")
    #     else:
    #         C.append("y")

    fig = plt.figure()
    ax = Axes3D(fig)

    #ax.plot_wireframe(X,Y,Z,rstride=1, cstride=1)

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

    #ax.scatter(X,Y,Z,alpha=0.4,s=10)

    #显示图像
    plt.show()
    pass

if __name__ == '__main__':
    base_count = np.zeros((632,472),np.float)
    plot_3d(base_count)