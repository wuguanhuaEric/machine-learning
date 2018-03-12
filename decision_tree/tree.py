#coding=utf-8
from math import log
import operator

#简单鱼分类
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

#计算香农熵 度量数据集无序程度
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for fearVec in dataSet:
        currentLabel = fearVec[-1]  #取最后一列键值 记录当前类别出现次数
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries  #该类别的概率
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

#划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[: axis]
            reducedFeatVec.extend(featVec[axis + 1:])  #extend方法是将添加元素融入集合
            retDataSet.append(reducedFeatVec)  #append将添加的元素作为一个元素加入
    return retDataSet

#选择最好的数据集划分
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0;bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]  #使用列表推导来创建新的列表
        uniqueVals = set(featList)  #以set数据类型保存，从列表中创建集合来获取列表中的唯一元素中
        newEntropy = 0.0
        for value in uniqueVals:  #遍历当前特征中的所有唯一属性值
            subDataSet = splitDataSet(dataSet, i, value)  #对每个特征划分数据集
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

#确定决策树叶子节点的分类
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

#创建决策树
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:  #使用完了所有的特征
        return majorityCnt(classList)  #返回出现次数最多的特征
    #创建树
    bestFeat = chooseBestFeatureToSplit(dataSet)  #将选取的最好特征放在bestFeat中
    bestFeatLabel = labels[bestFeat]  #特征标签
    myTree = {bestFeatLabel : {}}  #使用特征标签创建树
    del(labels[bestFeat])  #del用于list列表操作，删除一个或者连续几个元素
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

#使用决策树来分类
def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]  #找到输入的第一个元素
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)  #建立索引
    for key in secondDict.keys():
        if testVec[featIndex] == key:  #若该特征值等于当前key，往下走
            if type(secondDict[key]).__name__ == 'dict':  #若为树结构
                classLabel = classify(secondDict[key], featLabels, testVec)  #递归调用
            else: classLabel = secondDict[key]  #为叶子节点，赋予label值
    return classLabel  #返回分类结果

#决策树的存储
def storeTree(inputTree, filename):  #序列化的对象可以在磁盘上保存，需要时读取
    import pickle  #Python序列化对象，保存树结构的字典对象
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):  #读取对象
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


#测试样例
dataSet, labels = createDataSet()
myTree = createTree(dataSet, labels) #会改变labels集合
dataSet, labels = createDataSet()
testVec = [1, 1]
print classify(myTree, labels, testVec)
testVec2 = [1, 0]
print classify(myTree, labels, testVec2)

