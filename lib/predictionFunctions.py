import torch
import torch.nn.functional as F

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from collections import Counter

def predict(clasifier, tokens, categories):
    with torch.no_grad():
        outputs = clasifier(**tokens)
    logits = outputs.logits
    label = logits.argmax().item()
    probabilities = F.softmax(logits, dim=-1)
    return probabilities

def getMispredictedClasses(prediction, trueValue):
    assert prediction.shape == trueValue.shape
    mask = prediction != trueValue
    return prediction[mask]

def getMispredictedCategories(prediction, trueValue, categories):
    wrong = [categories[i] for i in getMispredictedClasses(prediction, trueValue).tolist()]
    groundTruth = [categories[i] for i in getMispredictedClasses(trueValue, prediction).tolist()]
    return [(w, g) for w, g in zip(wrong, groundTruth)]

def getCategoryBasedAccuracy(prediction, trueValue, categories):
    mispredistedCategories = getMispredictedCategories(prediction, trueValue, categories)
    countMispredictedCategories = dict(Counter([item for (_, item) in mispredistedCategories]))
    countTotalCategories = dict(Counter([categories[item] for item in trueValue.tolist()]))
    missingItems = {item:0 for item in set(countTotalCategories.keys()).difference(set(countMispredictedCategories.keys()))}
    countMispredictedCategories = {**countMispredictedCategories, **missingItems}
    return {cat:(1 - (countMispredictedCategories[cat]/countTotalCategories[cat])) for cat in countTotalCategories.keys()}
    
def getMetrics(probsLabelsTuple):
    predictedProbabilities, expected = probsLabelsTuple
    predictions = predictedProbabilities.argmax(axis=-1)
    accuracy = accuracy_score(expected, predictions)
    # Due to the imbalace of classes, let's get also the F1 scores (micro, macro and weighted)
    f1Macro = f1_score(expected, predictions, average='macro')
    f1Micro = f1_score(expected, predictions, average='micro')
    f1Weighed = f1_score(expected, predictions, average='weighted')
    return {"Accuracy":accuracy, "F1Macro":f1Macro, "F1Micro":f1Micro, "F1Weighed":f1Weighed}
