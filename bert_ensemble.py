import torch
import lib.datasetFunctions as datasetFunctions
import lib.predictionFunctions as predictionFunctions
import torch.nn.functional as F

from lib.balancedDatasetExtractor import BalancedDatasetExtractor
from lib.questionsDataset import QuestionsDataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import f1_score
from collections import Counter

bertName = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bertName)

categories = ['Discovery', 'Troubleshooting', 'Code', 'Comparison', 'Advice', 'Off-topic']
categoriesDict = {label:i for i, label in enumerate(categories)}
numberOfClasifiers = 15

allData = datasetFunctions.getDataAsList('data.json')
datasetFunctions.cleanData(allData, categories)
labelsToData = datasetFunctions.getLabelsToData(allData, categories)
datasetFunctions.shuffleLabelsToData(labelsToData)
trainLabelsToData, validationLabelsToData = datasetFunctions.splitLabelsToData(labelsToData, 0.9)

print()
print('All data:', datasetFunctions.getLabelsToDataLength(labelsToData))
print('Train (and test) data:', datasetFunctions.getLabelsToDataLength(trainLabelsToData))
print('Final validation data:', datasetFunctions.getLabelsToDataLength(validationLabelsToData))
print()

validationDataSize = sum(datasetFunctions.getLabelsToDataLength(validationLabelsToData).values())
xVal, yVal = datasetFunctions.getInOutLabelsToData(validationLabelsToData, categoriesDict)
xValTokens = tokenizer(xVal, padding="max_length", truncation=True, max_length=512, return_tensors="pt").to(torch.device('cuda'))
print('expected validation labels: ', yVal)

allProbabilities = torch.zeros(numberOfClasifiers, validationDataSize, len(categories))
for i, datasetDict in enumerate(BalancedDatasetExtractor(trainLabelsToData, numberOfClasifiers)):
    train, test = datasetFunctions.splitLabelsToData(datasetDict, 0.9)
    xTrain, yTrain = datasetFunctions.getInOutLabelsToData(train, categoriesDict)
    xTest, yTest = datasetFunctions.getInOutLabelsToData(test, categoriesDict)

    xTrainTokens = tokenizer(xTrain, padding="max_length", truncation=True, max_length=512)
    xTestTokens = tokenizer(xTest, padding="max_length", truncation=True, max_length=512)
    trainDataset = QuestionsDataset(xTrainTokens, yTrain)
    testDataset = QuestionsDataset(xTestTokens, yTest)

    clasifier = BertForSequenceClassification.from_pretrained(bertName, num_labels=len(categories))

    trainArgs = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=20,
        learning_rate=5e-5,
        weight_decay=0.01,
        load_best_model_at_end = True
    )

    trainer = Trainer(
        model=clasifier,
        args=trainArgs,
        train_dataset=trainDataset,
        eval_dataset=testDataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
        compute_metrics=predictionFunctions.getMetrics
    )

    trainer.train()

    allProbabilities[i] = predictionFunctions.predict(clasifier, xValTokens, categories)


expected = torch.tensor(yVal)
torch.save(allProbabilities, "allPredictedProbabilities.pth")
torch.save(expected, "expected.pth")

print('allProbabilities.shape:', allProbabilities.shape)

# summing up probabilities predicted by each model in the ensemble: equivalent to the mean probability of the ensemble (dividing the sum by the same number)
sumOfProbabilitiesOfAllModels = torch.sum(allProbabilities, dim=0)
print('\nsumOfProbabilitiesOfAllModels: ', sumOfProbabilitiesOfAllModels.shape)

maxSumOfProbabilities = torch.argmax(sumOfProbabilitiesOfAllModels, dim=1)
print('\nmaxSumOfProbabilities', maxSumOfProbabilities.shape, maxSumOfProbabilities)

expectedVsmaxSumOfProbabilities = torch.sum(maxSumOfProbabilities != expected).item()
print('\nMismatches between expected and maxSumOfProbabilities:', expectedVsmaxSumOfProbabilities)

majorityVoting = torch.argmax(allProbabilities, dim=2)
print('\nmajorityVoting', majorityVoting.shape)

majorityVoting, _ = torch.mode(majorityVoting, dim=0)
print('\nmajorityVoting', majorityVoting.shape, majorityVoting)

expectedVsmajorityVoting = torch.sum(majorityVoting != expected).item()
print('\nMismatches between expected and majorityVoting:', expectedVsmajorityVoting)

print('\nMismatches between maxSumOfProbabilities and majorityVoting:', torch.sum(majorityVoting != maxSumOfProbabilities).item())

print('\nExpected:', expected.shape, expected)

print('\nDiff at: maxSumOfProbabilities vs. expected:', predictionFunctions.getMispredictedCategories(maxSumOfProbabilities, expected, categories))
print('\nDiff at: majorityVoting vs. expected:', predictionFunctions.getMispredictedCategories(majorityVoting, expected, categories))
print('\nDiff at: majorityVot vs. maxSumOfProbabilities:', predictionFunctions.getMispredictedCategories(majorityVoting, maxSumOfProbabilities, categories))

predictions = maxSumOfProbabilities
mispredistedCategories = predictionFunctions.getMispredictedCategories(predictions, expected, categories)
print('\nMispredicted categories as tuples (predicted, expected) in validation set:', mispredistedCategories)
categoryBasedAccuracy = predictionFunctions.getCategoryBasedAccuracy(predictions, expected, categories)
print('\nCategory based accuracy in validation set:', categoryBasedAccuracy)
print('\nCategories count in the validation set:', dict(Counter([categories[i] for i in yVal])))

print('\nOverall accuracy:', 1 - (expectedVsmaxSumOfProbabilities/maxSumOfProbabilities.shape[0]))
