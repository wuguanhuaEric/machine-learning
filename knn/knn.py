#coding=utf-8
from numpy import *  #科学计算包numpy
import operator      #运算符模块
import os, sys

#k-近邻算法

#创建数据集
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

#计算距离
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  #shape读取数据矩阵第一维度的长度
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  #tile重复数组inX,有dataSet行，1个dataSet列，减去计算差值
    sqDiffMat = diffMat**2  #利用欧氏距离
    sqDistances = sqDiffMat.sum(axis=1)  #axis = 0为普通相加，axis=1为一行的行向量相加
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()  #返回数组从小到大的索引值
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]  #根据排序结果的索引值返回靠近的前k个标签
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  #各个标签出现的频率
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)

    return sortedClassCount[0][0]  #找出频率最高的


def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)  #读出数据行数
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()  #删除空白符
        listFromLine = line.split('\t')  #split指定分隔符对数据切片
        returnMat[index, :] = listFromLine[0:3]  #选取前3个元素（特征存储在返回矩阵中）
        classLabelVector.append(int(listFromLine[-1]))  #最后一列信息存储在classLabelVector
        index += 1
    return returnMat, classLabelVector

#归一化特征值
#归一化公式：（当前值-最小值）/range
def autoNorm(dataSet):
    minVals = dataSet.min(0)  #存放每列最小值，参数0使得可以从列中选取最小值，而不是当前行
    maxVals = dataSet.max(0)  #存放每列最大值
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))  #初始化归一化矩阵为读取的dataSet
    m = dataSet.shape[0]  #m保存第一行
    #特征矩阵是3 * 1000， min max range均为1*3，因此采用tile将变量内容复制成输入矩阵同大小
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m ,1))
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.10  #测试样本所占比例
    datingDataMat, datingLabels = file2matrix('test2.txt')  #读取文件
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is %d" % (classifierResult, datingLabels[i]))
        if(classifierResult != datingLabels[i]):errorCount += 1.0
    print errorCount
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))

def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(raw_input("percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year"))
    datingDataMat, datingLabels = file2matrix('test2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("You will probably like this person:",  resultList[classifierResult - 1])


def img2vector(filename):
    returnVect = zeros((1, 1024));  #每个手写识别为32*32大小的二进制图像矩阵  转换为1*1024 numpy向量数组returnVect
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])  #将每行的32 个字符值存储在numpy数组中
    return returnVect



#测试算法
def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir('trainingDigit')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))  #定义文件数x每个向量的训练集
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  #解析文件
        classNumStr = int(fileStr.split('_')[0])  #解析文件名
        hwLabels.append(classNumStr)  #存储类别
        trainingMat[i, :] = img2vector('trainingDigits/%s'% fileNameStr)
    #测试数据集
    testFileList = os.listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s'%fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with:%d, the real answer is:%d" %(classifierResult, classNumStr))
        if(classifierResult != classNumStr):
            errorCount += 1.0
        print("\nthe total number of error is: %d" % errorCount)
        print("\nthe total rate is: %f" %(errorCount/float(mTest)))
