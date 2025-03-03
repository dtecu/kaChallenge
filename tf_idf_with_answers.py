import numpy as np
import lib.datasetFunctions as datasetFunctions
import lib.predictionFunctions as predictionFunctions
import lib.tfIdfFunctions as tfIdfFunctions

from sklearn.metrics import f1_score

categories = ['Discovery', 'Troubleshooting', 'Code', 'Comparison', 'Advice', 'Off-topic']
categoriesDict = {label:i for i, label in enumerate(categories)}

allData = datasetFunctions.getDataAsList('data.json')
datasetFunctions.cleanData(allData, categories)

labelsToData = datasetFunctions.getLabelsToData(allData, categories)
datasetFunctions.shuffleLabelsToData(labelsToData)
trainLabelsToData, validationLabelsToData = datasetFunctions.splitLabelsToData(labelsToData, 0.8)

trainLabelsToData = tfIdfFunctions.concatenateDocsInSameCategoryWithAnswers(trainLabelsToData)
trainDocuments = tfIdfFunctions.getListOfCategories(trainLabelsToData, categories)
vectorizer, matrix = tfIdfFunctions.createTfidfMatrix(trainDocuments)

validationDocsDict = tfIdfFunctions.getListOfDocsForLabel(validationLabelsToData)

allCorrectPredictions, allPredictions = 0, 0
for categ in categories:
    predicted = tfIdfFunctions.compareNewDocuments(validationDocsDict[categ], vectorizer, matrix)
    correctlyPredicted = np.count_nonzero(predicted == categoriesDict[categ])
    allCorrectPredictions += correctlyPredicted
    allPredictions += len(predicted)
    print('Accuracy for', categ, ':', correctlyPredicted/len(predicted))
print('Overall accuracy:', allCorrectPredictions/allPredictions)
