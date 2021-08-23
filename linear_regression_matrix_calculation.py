# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 03:17:29 2021

@author: mohammed batis - 18511160002
"""

import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    numFeat=len(open(fileName).readline().split('\t'))-1
    dataMat=[]
    labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

xArr,yArr = loadDataSet("test.txt")

X = np.array(xArr)
y = np.transpose(np.array(yArr))

# Compute beta
Xt = np.transpose(X)
XtX = np.dot(Xt,X)
Xty = np.dot(Xt,y)
beta = np.linalg.solve(XtX,Xty)
print(beta,"\n")

for data,actual in zip(xArr,yArr):
    x = np.array([data])
    prediction = np.dot(x,beta)
    print('prediction = '+str(prediction)+', actual = '+str(actual)+", Error =",prediction.astype(np.float)-actual)
