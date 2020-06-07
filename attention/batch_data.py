import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from protein_loader import ProteinLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BatchData:
    def __init__(self, input_path, batch_size, n_workers, validation_split, shuffle_dataset):
        self.input_path = input_path
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.validation_split = validation_split
        self.shuffle_dataset = shuffle_dataset

    def dataset(self):
        self.dataset = ProteinLoader(self.input_path)
        return self.dataset

    def batch_data(self):
        random_seed = 42
        dataset_size = len(self.dataset) # size of datasets to be analysed

        # Creating data indices for training and validation splits:
        indices = list(range(dataset_size))
        split = int(np.floor(self.validation_split * dataset_size))

        if self.shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        # dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)

        self.train_loader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=self.batch_size,
                                                   num_workers=self.n_workers,
                                                   sampler=train_sampler)
        self.validation_loader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=self.batch_size,
                                                        num_workers=self.n_workers,
                                                        sampler=valid_sampler)

        return self.train_loader, self.validation_loader


"""
# Print to test ouputs
print('=' * 83)
dataiter = iter(train_loader)
data = dataiter.next()
features, labels = data
print(f"features: \n{features}\n"
      f"labels: \n{labels}\n"
      f"len features: {len(features)}\n"
      f"len labels: {len(labels)}\n")
print('=' * 83)
dataiter = iter(validation_loader)
data = dataiter.next()
features, labels = data
print(f"validation features: \n{features}\n"
      f"validation labels: \n{labels}\n"
      f"len validation features: {len(features)}\n"
      f"len validation labels: {len(labels)}\n")
print('=' * 83)
print(len(train_loader))
print(len(validation_loader))
print('=' * 83)
"""