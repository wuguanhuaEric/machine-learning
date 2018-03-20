#coding=utf-8
from numpy import *

#load data
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():  #for each line
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat

#distance func
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

#init K points randomly
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * rand(k, 1)
    return centrids

#K-均值算法：
def kMeans(dataSet, k, distMeas = distEclud, createCent = randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zero((m, 2)))  #store the result matrix, 2 cols for index and error
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  #for every points
            minDist = inf; minIndex = -1  #init
            for j in range(k):  #for every k centers, find the nearest center
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI;minIndex = j  #update distance and index
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True  #判断数据点所属类别与之前是否相同，只要有一个点变化就重设为True,再次迭代
            clusterAssment[i, :] = minIndex, minDist ** 2
        print(centroids)
        #update k center
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A==cent)[0]]  #nonzero()函数将满足条件的索引值筛选出来，().A是将matrix转化为array，因为nonzero()函数返回的是二维的元组，第一项为索引数组，第二项为类型，所以用[0]来确定
            centroids[cent, :] = mean(ptsInClust, axis = 0)
    return centroids, clusterAssment

#二分k-均值聚类
def biKmeans(dataSet, k, distMeas = distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroid0 = mean(dataSet, axis = 0).tolist()[0]
    centList = [centroid0]  #create a list with one centroid
    for j in range(m):  #calc initial Error for each point
        clusterAssment[j, 1] = disMeas(mat(centroid0), dataSet[j, :]) ** 2
    while (len(centList) < k):
        lowestSSE = inf  #init SSE
        for i in range(len(centList)):  #for every centroid
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]  #get the data points currently in cluster i
        centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)  #k = 2
        sseSplit = sum(splitClustAss[:, 1])  #compare the SSE to the current minimum
        sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
        print("sseSplit, and notSplit:" , sseSplit, sseNotSplit)
        if (sseSplit + sseNotSplit) < lowestSSE:  #judge the error
            bestCentToSplit = i
            bestNewCents = centroidMat
            bestClustAss = splitClustAss.copy()
            lowestSSE = sseSplit + sseNotSplit
        #new cluster and split cluster
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)  #change 1 to 3, 4 or whatever
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print('the bestCentToSplit is: ', bestCentToSplit)
        print('the len of bestClusterAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]  #replace a centroid with two best centroids
        centList.append(bestNewCents[1, :].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss  #reassign new clusters, and SSE
     return mat(centList), clusterAssment
