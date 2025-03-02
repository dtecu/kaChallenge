import torch
import torch.nn.functional as F

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
