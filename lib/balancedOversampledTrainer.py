import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer

class BalancedOversampledTrainer(Trainer):
    def __init__(self, y, **kwargs):
        super().__init__(**kwargs)
        self.y = y
        self.dataLoader = self.createBalancedOversampledDataLoader()

    def createBalancedOversampledDataLoader(self):
        categoryCounts = torch.bincount(self.y)
        weights = 1. / categoryCounts.float()
        sampleWeights = weights[self.y]
        sampler = torch.utils.data.WeightedRandomSampler(sampleWeights, len(sampleWeights))
        dataset = torch.utils.data.TensorDataset(self.x, self.y)
        return DataLoader(self.train_dataset, batch_size=8, sampler=sampler)

    def get_train_dataloader(self):
        return self.dataLoader
