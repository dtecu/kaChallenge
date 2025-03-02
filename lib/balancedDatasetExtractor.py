import lib.datasetFunctions

class BalancedDatasetExtractor:
    def __init__(self, labelsToData, datasetsNumber):
        self.labelsToData = labelsToData
        self.dataPerLabelSize = min(datasetFunctions.getLabelsToDataLength(labelsToData).values())
        self.current = 0
        self.end = datasetsNumber

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.end:
            raise StopIteration
        result = self.getLabelsToData()
        self.current += 1
        return result
    
    def getLabelsToData(self):
        result = dict()
        start = self.current * self.dataPerLabelSize
        end = (self.current + 1) * self.dataPerLabelSize
        for label, data in self.labelsToData.items():
            i = start % len(data)
            j = (end - 1) % len(data) + 1
            if i <= j:
                result[label] = data[i:j]
            else:
                result[label] = data[i:] + data[:j]
            assert len(result[label]) == self.dataPerLabelSize, 'Problem when generating dataset'
        return result
