import json
import random
import torch
from functools import reduce

RANDOM_SEED = 123
random.seed(RANDOM_SEED)

def getDataAsList(fileName):
    with open(fileName, 'r') as file:
        allData = json.load(file)['data']
    return allData

def correctMislabelledData(data):
    for item in data:
        if item['category'] == 'Off-Topic':
            item['category'] = 'Off-topic'

def cleanData(allData, categories):
    print('Length of all data in data set:', len(allData))
    categoryToSampleSize = {label:len([data['category'] for data in allData if data['category'] == label]) for label in categories}
    print('Size of categories in the data set:', categoryToSampleSize)
    labelledSize = reduce(lambda x, y: x+y, categoryToSampleSize.values(), 0)
    if labelledSize != len(allData):
        print('There are %d unlabelled items' % (len(allData) - labelledSize,))
        print('Figuring out why ...')
        unlablelledData = [data['category'] for data in allData if data['category'] not in categories]
        print(unlablelledData)
        print("OK the reason is the inconsistency: 'Off-Topic' vs 'Off-topic'")
        print("Agreeing to turn all of 'Off-Topic' into 'Off-topic' and then check again that the data is clean")
        correctMislabelledData(allData)
        print('Confirm that data is consistent')
        print('Length of all data in data set:', len(allData))
        categoryToSampleSize = {label:len([data['category'] for data in allData if data['category'] == label]) for label in categories}
        print('Size of categories in the data set:', categoryToSampleSize)
        labelledSize = reduce(lambda x, y: x+y, categoryToSampleSize.values(), 0)
        assert labelledSize == len(allData), 'There are %d unlabelled items' % (len(allData) - labelledSize,)
    print('Data is OK now')

def getLabelsToData(allData, categories):
    return {label:[data for data in allData if data['category'] == label] for label in categories}

def getLabelsToDataLength(labelsToData):
    return {label:len(data) for label,data in labelsToData.items()}

def shuffleData(data):
    random.shuffle(data)

def shuffleLabelsToData(labelsToData):
    for label, data in labelsToData.items():
        shuffleData(data)

def splitData(data, pTrain):
    n = len(data)
    return data[:int(n * pTrain)], data[int(n * pTrain):]

def splitLabelsToData(labelsToData, pTrain):
    trainLabelsToData = dict()
    testLabelsToData = dict()
    for label, data in labelsToData.items():
        trainData, testData = splitData(data, pTrain)
        trainLabelsToData[label] = trainData
        testLabelsToData[label] = testData
    return trainLabelsToData, testLabelsToData

def getInOutData(data, categoriesDict):
    input = [item['question'] for item in data]
    output = [categoriesDict[item['category']] for item in data]
    return input, output

def getInOutLabelsToData(labelsToData, categoriesDict, shuffle=True):
    allData = []
    for _, data in labelsToData.items():
        allData.extend(data)
    if shuffle:
        shuffleData(allData)
    input = [item['question'] for item in allData]
    output = [categoriesDict[item['category']] for item in allData]
    return input, output
