import random
import lib.datasetFunctions

categories = ['Discovery', 'Troubleshooting', 'Comparison', 'Off-topic']
categoriesDict = {label:i for i, label in enumerate(categories)}

expectedAllData = [{'category': 'Troubleshooting', 'question': 0}, {'category': 'Discovery', 'question': 1}, {'category': 'Discovery', 'question': 2}, {'category': 'Off-topic', 'question': 3}, {'category': 'Discovery', 'question': 4}, {'category': 'Troubleshooting', 'question': 5}, {'category': 'Comparison', 'question': 6}, {'category': 'Discovery', 'question': 7}, {'category': 'Off-topic', 'question': 8}, {'category': 'Discovery', 'question': 9}, {'category': 'Discovery', 'question': 10}, {'category': 'Comparison', 'question': 11}]

expectedLabelsToData = {'Discovery': [{'category': 'Discovery', 'question': 1}, {'category': 'Discovery', 'question': 2}, {'category': 'Discovery', 'question': 4}, {'category': 'Discovery', 'question': 7}, {'category': 'Discovery', 'question': 9}, {'category': 'Discovery', 'question': 10}], 'Troubleshooting': [{'category': 'Troubleshooting', 'question': 0}, {'category': 'Troubleshooting', 'question': 5}], 'Comparison': [{'category': 'Comparison', 'question': 6}, {'category': 'Comparison', 'question': 11}], 'Off-topic': [{'category': 'Off-topic', 'question': 3}, {'category': 'Off-topic', 'question': 8}]}

expectedLabelsToDataLength = {'Discovery':6, 'Troubleshooting':2, 'Comparison':2, 'Off-topic':2}

expectedLabelsToDataShuffled = {'Discovery': [{'category': 'Discovery', 'question': 7}, {'category': 'Discovery', 'question': 9}, {'category': 'Discovery', 'question': 2}, {'category': 'Discovery', 'question': 10}, {'category': 'Discovery', 'question': 4}, {'category': 'Discovery', 'question': 1}], 'Troubleshooting': [{'category': 'Troubleshooting', 'question': 5}, {'category': 'Troubleshooting', 'question': 0}], 'Comparison': [{'category': 'Comparison', 'question': 11}, {'category': 'Comparison', 'question': 6}], 'Off-topic': [{'category': 'Off-topic', 'question': 3}, {'category': 'Off-topic', 'question': 8}]}

expectedSplitLabelsToData = ({'Discovery': [{'category': 'Discovery', 'question': 7}, {'category': 'Discovery', 'question': 9}, {'category': 'Discovery', 'question': 2}], 'Troubleshooting': [{'category': 'Troubleshooting', 'question': 5}], 'Comparison': [{'category': 'Comparison', 'question': 11}], 'Off-topic': [{'category': 'Off-topic', 'question': 3}]}, {'Discovery': [{'category': 'Discovery', 'question': 10}, {'category': 'Discovery', 'question': 4}, {'category': 'Discovery', 'question': 1}], 'Troubleshooting': [{'category': 'Troubleshooting', 'question': 0}], 'Comparison': [{'category': 'Comparison', 'question': 6}], 'Off-topic': [{'category': 'Off-topic', 'question': 8}]})

expectedInOut = ([7, 2, 9, 11, 5, 3], [0, 0, 0, 2, 1, 3])

allData = datasetFunctions.getDataAsList('test.json')
assert allData == expectedAllData

labelsToData = datasetFunctions.getLabelsToData(allData, categories)
assert labelsToData == expectedLabelsToData

assert datasetFunctions.getLabelsToDataLength(labelsToData) == expectedLabelsToDataLength
assert labelsToData == expectedLabelsToData

random.seed(123)
datasetFunctions.shuffleLabelsToData(labelsToData)
assert labelsToData == expectedLabelsToDataShuffled
assert expectedLabelsToData != expectedLabelsToDataShuffled

trainData, testData = datasetFunctions.splitLabelsToData(labelsToData, 0.5)
assert (trainData, testData) == expectedSplitLabelsToData
assert trainData == expectedSplitLabelsToData[0]
assert testData == expectedSplitLabelsToData[1]
assert labelsToData == expectedLabelsToDataShuffled

random.seed(789)
assert datasetFunctions.getInOutLabelsToData(trainData, categoriesDict, shuffle=True) != expectedInOut
random.seed(456)
assert datasetFunctions.getInOutLabelsToData(trainData, categoriesDict, shuffle=True) == expectedInOut
assert trainData == expectedSplitLabelsToData[0]
assert labelsToData == expectedLabelsToDataShuffled

print('Tests pass')
