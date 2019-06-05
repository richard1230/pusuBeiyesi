

from numpy import *

def loadDataSet():
    postingList = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]#0不是脏话,1是脏话
    return postingList,classVec

postingList, classVec = loadDataSet()


#去重复并合并整个文档
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        # print (len(document))                   #7--->8--->...

        # print(document)                        #['my', 'dog', 'has', 'flea', 'problems', 'help', 'please']--->['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid']--->...
        vocabSet = vocabSet | set(document)  # 集合的并集
    # print(len(vocabSet))                        #32
    # print("合并文档函数")
    return list(vocabSet)

vocabList = createVocabList(postingList)
# print(vocabList)
#32个单词
#['steak', 'to', 'is', 'licks', 'has', 'love', 'dog', 'park', 'my', 'help', 'dalmation', 'stupid', 'flea', 'take', 'him', 'so', 'I', 'how', 'problems', 'cute', 'maybe', 'garbage', 'posting', 'ate', 'quit', 'mr', 'please', 'worthless', 'food', 'not', 'buying', 'stop']


#单词转换为向量

def setOfWords2Vec(vocabList,inputSet):
    # print(vocabList)
    returnVec = [0] * len(vocabList)    #初始化
    # retest = [0]
    # print(len(vocabList))  #32
    # print(retest)       #[0]
    # print(returnVec)      #32个0:[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for word in inputSet:
        # print(word)
        returnVec[vocabList.index(word)] =1 #如果输入的inputSet里面的单词在上面的已经合并的文档里,那么则把在里面的响应的单词输出为1
    # print(returnVec)
    #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]
    # print("单词转向量函数")
    return returnVec


trainMatrix = []
myVocabList = createVocabList(postingList)
# print(myVocabList)
##['steak', 'park', 'my', 'stop', 'not', 'to', 'maybe', 'worthless', 'ate', 'him', 'stupid', 'dalmation', 'flea', 'food', 'how', 'garbage', 'so', 'has', 'dog', 'licks', 'quit', 'help', 'problems', 'cute', 'take', 'I', 'mr', 'posting', 'is', 'buying', 'love', 'please']
# print("----------")#上面32个单词


for postinDoc in postingList:
    # print("lalala啦啦啦")
    # print(postinDoc)
    # print("---开始了---")
    trainMatrix.append(setOfWords2Vec(myVocabList,postinDoc))    #myVocabList为合并后的32个单词
# print(trainMatrix)
#[[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1], [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]
#这里的意思是把输入数据的每一行(这里总共有6行)转换成向量的形式,即:如果单词是在合并的文档里面的话,就会为转成 1,输出的列表有6个元素,每个元素是一个列表，每个列表里面有32个元素,其中有每行数据元素的单词为1

## trainNB0 这个函数的作用:
#输入：单词的向量(),这里的意思是把输入数据的每一行(这里总共有6行)转换成向量的形式,即:如果单词是在合并的文档里面的话,就会为转成 1,输出的列表有6个元素,每个元素是一个列表，每个列表里面有32个元素,其中有每行数据元素的单词为1
#输入:种类(就是(0,1,0,1,0,1))
#输出:有脏话时特征对应的概率,无脏话时特征对应的概率,脏话的概率(P(B1)),无脏话对应的概率=1-有脏话时对应的概率
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)                     #这里为6,6个样本,计算出样本的个数
    numWords = len(trainMatrix[0])                      #表示单词数目32，计算是单词的数目
    pAbusive = sum(trainCategory)/float(numTrainDocs)   #计算出脏话的概率,这里为0.5,即P(B1),在这个例子里面只有B0和B1
    # print("脏话的概率")
    # print(pAbusive)
    p0Num = ones(numWords)                              #非脏话的计数,这里是初始化,初始化为32个1
    # print("非脏话计数初始化")
    # print(p0Num)
    p1NUm = ones(numWords)                              #脏话计数,这里初始化为32个1
    # print(p1NUm)
    p0Denom = 2.0                                       #这里初始化p0Denom为2是为了方便下面的计算
    p1Denom = 2.0                                       #这个初始化p1Denom为2是为了方便下面的计算
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:                       #numTrainDocs表示有6个,i从0至5,trainCategory这里表示传入的也有6个，为1表示为脏话，0表示非脏话，这里是把脏话和非脏话分离开来,这里==1指的是脏话
            # print(trainMatrix[i])
            #[1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,0, 1, 0, 0, 0]   trainMatrix[1]
            #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,0, 0, 0, 1, 0, 1, 0, 0]   trainMatrix[3]
            #[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,0, 0, 0,0, 0, 1, 0, 0]  trainMatrix[5]
            # print("这里是trainMatrix[i]")
            p1NUm += trainMatrix[i]                     #这里展示出现的脏话，并计数,就是把trainMatrix[1] + trainMatrix[3] + trainMatrix[5]加起来,这样就把六个样本里面的所有脏话都给集合到一起了,同时将其向量化了, 总共32个单词,哪个位置上面出现了脏话，就➕加1
            # print(p1NUm)
            #[2. 1. 1. 1. 1. 1. 1. 1. 2. 2. 2. 1. 1. 2. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.2. 2. 1. 1. 2. 1. 1. 1.]   p1Num
            #[2. 1. 1. 1. 1. 1. 1. 1. 2. 2. 3. 1. 1. 2. 1. 1. 2. 2. 1. 1. 1. 1. 1. 1.2. 2. 1.2. 2. 2. 1. 1.]  p1Num
            #[2. 1. 1. 2. 1. 1. 1. 1. 2. 2. 4. 1. 1. 3. 2. 1. 2. 2. 1. 1. 1. 1. 2. 1.2. 2. 1.2. 2. 3. 1. 1.] p1Num
            # print("这里是p1Num")
            # print("-------------")

            p1Denom += sum(trainMatrix[i])              #这里为脏话出现的次数计数，(初始值，自己设定)+8=10.0---》15---》21
            # print(p1Denom)          #2(初始值，自己设定)+8=10.0---》15---》21
            # print("这里是p1Denom")

        else:
            p0Num += trainMatrix[i]                    #这里计数为非脏话
            p0Denom += sum(trainMatrix[i])             #这里统计非脏话出现的次数
    p1 = p1NUm/p1Denom                                  #这里为有脏话时特征对应的概率,即P(A1|B0),P(A2|B0)...P(A32|B0)
    p1Vect =log (p1NUm/p1Denom)                         #加Log是为了方便计算
    print(p1)
    print(p1Vect)
    p0 = p0Num/p0Denom                                 #这里计算非脏话时特征对应的的概率，即P(A1|B1),P(A2|B1)...P(A32|B1)
    p0Vect = log(p0Num/p0Denom)                        #加log是为了后面计算方便
    print(p0)
    print(p0Vect)
    return p0Vect,p1Vect,pAbusive

#[1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,0, 1, 0, 0, 0]   trainMatrix[1]
#
# [2. 1. 1. 1. 1. 1. 1. 1. 2. 2. 2. 1. 1. 2. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.2. 2. 1. 1. 2. 1. 1. 1.]   p1Num
#
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,0, 0, 0, 1, 0, 1, 0, 0]   trainMatrix[3]
#
# [2. 1. 1. 1. 1. 1. 1. 1. 2. 2. 3. 1. 1. 2. 1. 1. 2. 2. 1. 1. 1. 1. 1. 1.2. 2. 1.2. 2. 2. 1. 1.]  p1Num
#
# [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,0, 0, 0,0, 0, 1, 0, 0]  trainMatrix[5]
#
# [2. 1. 1. 2. 1. 1. 1. 1. 2. 2. 4. 1. 1. 3. 2. 1. 2. 2. 1. 1. 1. 1. 2. 1.2. 2. 1.2. 2. 3. 1. 1.] p1Num


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):#这里的vec2Classify是测试集
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)                               #p(Bi|A) = P(Bi)P(A1|Bi)P(A2|Bi)P(A3|Bi)/(P(A1)P(A2)P(A3));这里把分子上面的乘法转换成加法了，这里的P(Bi)就是trainNB0里面的pAbusive，pAbusive对应的就是这里的pClass1，vec2Classify指的是测试集,p1Vec乘以测试集就会得到P(A1|B1)*P(A2|B1)。。。P(A32|B1)，由于分母是一样的,故只需要比较P(B1|A)和P(B0|A)的分子即可
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)                         #这里是计算的P(B1|A)
    if p1>p0:                                                                   #这里是比较P(B0|A)和P(B1|A)的大小,哪个大就取哪个
        return 1
    else:
        return 0


##有关array的相关解释ß
# trainmat = [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1], [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]
# arr = array(trainmat)
# print(trainmat)
# #[[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1], [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]
#
# print(arr)
# [[0 0 0 0 1 0 0 0 0 0 1 0 1 0 0 0 0 0 1 0 0 0 1 1 0 1 0 0 0 0 0 0]
#  [0 1 0 0 0 0 0 0 0 0 1 1 0 0 0 1 1 0 0 0 0 0 0 0 1 0 1 1 0 0 0 0]
#  [0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 1 0 0 0 1 0 0 1 0 0 1 0 1]
#  [0 0 0 0 0 1 0 0 0 1 0 0 0 0 1 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0]
#  [1 0 1 0 0 1 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 1 0 0 1 0 0 0 1 0]
#  [0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0]]

def testingNB():
    listOPosts,listClasses = loadDataSet()                                      #加载数据,listOPosts为原来文本，总共6个列表
    myVocabList = createVocabList(listOPosts)                                   #合并文档，输出32个单词(把重复的单词去除掉)
    trainMat = []
    for postinDoc in  listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))                  ##这里的意思是把输入数据的每一行(这里总共有6行)转换成向量的形式,即:如果单词是在合并的文档里面的话,就会为转成 1,输出的列表有6个元素,每个元素是一个列表，每个列表里面有32个元素,其中有存在每行数据元素的单词，则这个位置为1
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))                  #

    testEntry = ['love','my','dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))                      #[0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0]
    print(testEntry, '类别为: ', classifyNB(thisDoc, p0V, p1V, pAb))             #
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, '类别为: ', classifyNB(thisDoc, p0V, p1V, pAb))

testingNB()

# print(loadDataSet())
# print(createVocabList(dataSet))
# setOfWords2Vec(vocabList,['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'])
# trainNB0(trainMatrix,classVec)
