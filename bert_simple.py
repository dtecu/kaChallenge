import torch
import lib.datasetFunctions as datasetFunctions
import lib.predictionFunctions as predictionFunctions
import torch.nn.functional as F

from lib.questionsDataset import QuestionsDataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from collections import Counter

bertName = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bertName)

categories = ['Discovery', 'Troubleshooting', 'Code', 'Comparison', 'Advice', 'Off-topic']
categoriesDict = {label:i for i, label in enumerate(categories)}

allData = datasetFunctions.getDataAsList('data.json')
datasetFunctions.cleanData(allData, categories)
datasetFunctions.shuffleData(allData)

# get the validation portion used for the final verdict (unseen data during training)
trainData, validationData = datasetFunctions.splitData(allData, 0.9)

# split the train data in train/test to be used during model training
trainData, testData = datasetFunctions.splitData(trainData, 0.9)

xTrain, yTrain = datasetFunctions.getInOutData(trainData, categoriesDict)
xTest, yTest = datasetFunctions.getInOutData(testData, categoriesDict)
xVal, yVal = datasetFunctions.getInOutData(validationData, categoriesDict)

xTrainTokens = tokenizer(xTrain, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
xTtestTokens = tokenizer(xTest, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
xValTokens = tokenizer(xVal, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

trainDataset = QuestionsDataset(xTrainTokens, yTrain)
testDataset = QuestionsDataset(xTtestTokens, yTest)
validationDataset = QuestionsDataset(xValTokens, yVal)

clasifier = BertForSequenceClassification.from_pretrained(bertName, num_labels=len(categories))

trainArgs = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=30,
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

# Get model evaluations on train, test and validation sets (according to metrics defined in predictionFunctions.getMetrics):
trainResults = trainer.evaluate(trainDataset)
testResults = trainer.evaluate(testDataset)
evaluationResults = trainer.predict(validationDataset)

# Saving predicted probabilities and expected results guarantees us we can re-calculate any metrics
# later without re-training or saving the model (bert-base-uncased has about 420MB)
assert torch.equal(torch.tensor(evaluationResults.label_ids), torch.tensor(yVal)), 'Unexpected problem'
predictedProbabilities = F.softmax(torch.tensor(evaluationResults.predictions), dim=-1)
torch.save(predictedProbabilities, "allPredictedProbabilities.pth")
torch.save(torch.tensor(yVal), "expected.pth")

print('\nTrain results:', trainResults)
print('\nTest results:', testResults)
print('\nFinal, validation results:', evaluationResults.metrics)

predictions = torch.argmax(predictedProbabilities, dim=-1)
mispredistedCategories = predictionFunctions.getMispredictedCategories(predictions, torch.tensor(yVal), categories)
print('\nMispredicted categories as tuples (predicted, expected) in validation set:', mispredistedCategories)
categoryBasedAccuracy = predictionFunctions.getCategoryBasedAccuracy(predictions, torch.tensor(yVal), categories)
print('\nCategory based accuracy in validation set:', categoryBasedAccuracy)
print('\nCategories count in the validation set:', dict(Counter([categories[i] for i in yVal])))
