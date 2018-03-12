#coding=utf-8
from numpy import *

def loadDataSet():
    dataMat=[];labelMat=[]
    fr=open('testSet.txt')
    for line in fr.readlines():
        lineArr=line.strip().split() #逐行读入并切分，每行的前两个值为X1，X2
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])#X0设为1.0，保存X1，X2
        labelMat.append(int(lineArr[2])) #每行第三个值对应类别标签
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

#gradient up function
def gradAscent(dataMatIn,classLabels):#100,3 matrix
    dataMatrix=mat(dataMatIn) #change to numpy matrix ,different features for col &sample for row
    labelMat=mat(classLabels).transpose()
    m,n=shape(dataMatrix)
    #parameter for train
    alpha=0.001 #step length
    maxCycles=500#iteration num
    weights=ones((n,1))
    for k in range(maxCycles):
        h=sigmoid(dataMatrix*weights) #h is a vector
        error=(labelMat-h) #compute the difference between real type and predict type
        weights=weights+alpha*dataMatrix.transpose()*error
    return weights #return the best parameter

#train SGA
def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)  #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

#improvement
def stocGradAscent1(dataMatrix, classLabels, numIter = 150):
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01  #alpha will descent as iteration rise, but does not be 0
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error *dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

#test logistic
def classifyVector(intX, weights):
    prob = sigmoid(sum(intX * weights))
    if prob > 0.5: return 1.0
    else: return 0.0


def colicTest():
    frTrain = open('horseTraining.txt')
    frTest = open('horseTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[-1]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10;errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iteration the average error rate is: %f" % (numTests, errorSum / float(numTests)))


multiTest()
