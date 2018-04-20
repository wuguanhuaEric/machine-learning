#coding=utf-8

from numpy import *

def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

#对数据进行分类
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#just classify the data
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray

#找到最佳决策树
def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m,1)))
    minError = inf #最小错误率，开始初始化为无穷大
    for i in range(n):#遍历数据集所有特征
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
        stepSize = (rangeMax-rangeMin)/numSteps #考虑数据特征，计算步长
        for j in range(-1, int(numSteps) + 1):  #遍历不同步长时的情况
            for inequal in ['lt', 'gt']:  #大于/小于阈值 切换遍历
                threshVal = (rangeMin + float(j) * stepSize) #设置阈值
                predictedVals = stumpClassify(dataMatrix, i, threshVal,inequal) #分类预测
                errArr = mat(ones((m, 1)))#初始化全部为1（初始化为全部不相等）
                errArr[predictedVals == labelMat] = 0#预测与label相等则为0，否则为1
                # 分类器与adaBoost交互
                # 权重向量×错误向量=计算权重误差（加权错误率）
                weightedError = D.T * errArr
                if weightedError < minError:
                    minError = weightedError #保存当前最小的错误率
                    bestClasEst = predictedVals.copy() #预测类别
                    #保存该单层决策树
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst #返回字典，错误率和类别估计

#完整adaboost算法
def adaBoostTrainDS(dataArr,classLabels,numIt=40): #numIt 用户设置的迭代次数
    weakClassArr = []
    m = shape(dataArr)[0]#m表示数组行数
    D = mat(ones((m,1))/m)   #初始化每个数据点的权重为1/m
    aggClassEst = mat(zeros((m,1)))#记录每个数据点的类别估计累计值
    for i in range(numIt):
        # 建立一个单层决策树，输入初始权重D
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
        print ("D:",D.T)
        # alpha表示本次输出结果权重
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))#1e-16防止零溢出
        bestStump['alpha'] = alpha  #alpha加入字典
        weakClassArr.append(bestStump)     #字典加入列表
        print ("classEst: ",classEst.T)
        # 计算下次迭代的新权重D
        expon = multiply(-1*alpha*mat(classLabels).T,classEst)
        D = multiply(D,exp(expon))
        D = D/D.sum()
        # 计算累加错误率
        aggClassEst += alpha*classEst
        print ("aggClassEst: ",aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m
        print ("total error: ",errorRate)
        if errorRate == 0.0: break#错误率为0时 停止迭代
    return weakClassArr,aggClassEst

#测试adaboost
def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)#待分类样例 转换成numpy矩阵
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):#遍历所有弱分类器
        classEst = stumpClassify(dataMatrix,\
                                 classifierArr[0][i]['dim'],\
                                 classifierArr[0][i]['thresh'],\
                                 classifierArr[0][i]['ineq'])
        aggClassEst += classifierArr[0][i]['alpha']*classEst
        print (aggClassEst) #输出每次迭代侯变化的结果
