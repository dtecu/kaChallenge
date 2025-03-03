import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def createTfidfMatrix(documents):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(documents)
    return vectorizer, matrix

def compareNewDocuments(anotherDocs, vectorizer, matrix):
    anotherVectors = vectorizer.transform(anotherDocs)
    similarityMatrix = cosine_similarity(anotherVectors, matrix)
    return np.argmax(similarityMatrix, axis=1)

def concatenateDocsInSameCategory(labelsToData):
    result = dict()
    for label, data in labelsToData.items():
        docs = [item['question'] for item in data]
        result[label] = ' '.join(docs)
    return result

def getListOfCategories(docsInSameCategory, categories):
    # make sure we keep the order in categories
    return [docsInSameCategory[category] for category in categories]
