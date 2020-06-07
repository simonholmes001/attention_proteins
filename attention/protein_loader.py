import os

import torch
from torch.utils.data import Dataset, DataLoader
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ProteinLoader(Dataset):

    def __init__(self, data_directory):
        super().__init__()
        self.data_directory = data_directory + '/'
        dataset = self.get_all_data()
        self.X = []
        self.y = []
        for i in range(len(dataset)):
            self.X.append(dataset[i][0])
            self.y.append((dataset[i][1]))
        self.n_samples = len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.n_samples

    def get_all_data(self):
        files_in_path = self.extract_name_prefixes_from_filenames()
        dataset = []
        for filename in files_in_path:
            dataset.append((self.get_features(filename), self.get_labels(filename)))
        return dataset

    def get_labels(self, filename):
        with open(self.data_directory + filename + 'labels.pickle', 'rb') as labels_file:
            labels = pickle.load(labels_file).float()
        return labels

    def get_features(self, filename):
        with open(self.data_directory + filename + 'features.pickle', 'rb') as features_file:
            features = pickle.load(features_file).float()
        return features

    def extract_name_prefixes_from_filenames(self):
        return set([self.reconstruct_filename(file) for file in self.get_data_filenames()])

    def get_data_filenames(self):
        """
        Makes a list of all the files in the directory
        :return:
        """
        return sorted([file for file in os.listdir(self.data_directory) if file.endswith(".pickle")])

    @staticmethod
    def reconstruct_filename(file):
        """
        Gets the initial part of the file name, extracts the PDB code from the filename
        :param file:
        :return:
        """
        return "_".join(file.split("_")[:-1]) + "_"

"""

dataset = ProteinLoader('/home/the_beast/mathisi/simon/attention/data')

# Print to test outputs
# for features, labels in dataset:
#     print(features)
#     print(labels)
# print(len(dataset))

# Print to test outputs
# data = dataset[3]
# features, labels = data
# print("Data: \{}".format(data))
# print("Features: \n{}\n Labels: \n{}".format(features, labels))

batch_size = 1
num_workers = 1
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# To investigate later, better way of iterating over the dataset??? TBD
# dataiter = for i,  in enumerate(dataset):

# Print to test ouputs
# dataiter = iter(dataloader)
# data = dataiter.next()
# features, labels = data
# print(features)

# Dummy training loop
num_epochs = 10
total_samples = len(dataset)
n_iterations = math.ceil(total_samples / batch_size)
print(total_samples, n_iterations)

for epoch in range(num_epochs):
    # print(f"{epoch}")
    for i, (inputs, labels) in enumerate(dataloader):
        # print(f"{i}")
        # forward pass, backward pass, update weights
        print(f"epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {labels.shape}")
"""