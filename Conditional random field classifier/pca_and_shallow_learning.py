import os
import numpy as np
import json
import matplotlib.pyplot as plt
import torch
from torch.utils.data import  DataLoader, IterableDataset
from torch.nn import functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
import argparse
from sklearn.metrics import classification_report
from plotting_results import *

COMBINATIONS_WITH_N = ['AAN', 'ATN', 'AGN', 'ACN', 'ANA', 'ANT', 'ANG', 'ANC', 'ANN', 'TAN', 'TTN', 'TGN', 'TCN', 'TNA', 'TNT', 'TNG', 'TNC', 'TNN', 'GAN', 'GTN', 'GGN', 'GCN', 'GNA', 'GNT', 'GNG', 'GNC', 'GNN', 'CAN', 'CTN',
                       'CGN', 'CCN', 'CNA', 'CNT', 'CNG', 'CNC', 'CNN', 'NAA', 'NAT', 'NAG', 'NAC', 'NAN', 'NTA', 'NTT', 'NTG', 'NTC', 'NTN', 'NGA', 'NGT', 'NGG', 'NGC', 'NGN', 'NCA', 'NCT', 'NCG', 'NCC', 'NCN', 'NNA', 'NNT', 'NNG', 'NNC', 'NNN']


class FileIterator:
    def __init__(self, directory):
        self.file_paths = [os.path.join(directory, file)
                           for file in sorted(os.listdir(directory))]
        self.file_iter = iter(self.file_generator())

    def file_generator(self):
        for file_path in self.file_paths:
            with open(file_path, 'r') as file:
                print(file_path)
                for line in file:
                    yield line

    def __iter__(self):
        return self.file_iter


class GeneDataset(IterableDataset):
    def __init__(self, directory, max_sequence_length=4000, pca_components=2):
        self.file_iterator = FileIterator(directory)
        self.max_sequence_length = max_sequence_length
        self.pca = PCA(n_components=pca_components)


    def __iter__(self):
        batch_count = 0
        for line in self.file_iterator:
            try:
                data_list = json.loads(line)
                features_list = []
                target_list = []
                batch_count +=1
                genes_in_sequence = 0
                mask_list = []  # List to store the mask tensors
                for data in data_list:
                    target = None
                    features = None
                    features = {k: parse_number(v) for k, v in data.items() if k != 'gene'}
                    # forgot to add "N" as base option, so adding all combinations with N if they are not present
                    features = add_combinations_with_N(features)
                    if len(features) != 129:
                        print(f"odd number of features in token: {len(features)}")
                        continue
                    target = int(data.get('gene', None))
                    if target == 1:
                        genes_in_sequence +=1
                    if features and target is not None:
                        features_list.append(
                            [f for f in features.values()])
                        target_list.append([target])
                if len(target_list)<self.max_sequence_length:
                    print("Small fragment")
                    continue
                self._transform_and_plot(
                        features_list, target_list, batch_count)
                print (f"Genes in sequence: {genes_in_sequence}")
                yield features_list, target_list  # Return the mask tensor

    def _transform_and_plot(self, features, targets, batch_count):
        try:
            transformed_features = self.pca.fit_transform(features)
            plt.figure(figsize=(8, 6))
            plt.scatter(transformed_features[:, 0],
                    transformed_features[:, 1], c=targets)
            plt.colorbar()
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.title(f'PCA of data for batch {batch_count}')
            plt.savefig(os.path.join(output_directory, "pca_plots", f"pca_batch_{batch_count}.png"))
            plt.close()
        
            return transformed_features
        except:
            print(features)

    def _parse_number(self, s):
        # If the string starts with '0.', don't strip the leading zero
        if not isinstance(s, str):
            return s
        if s.startswith('0.'):
            num = float(s)
        else:
            stripped = s.lstrip('0')

            # Check if the stripped string is empty (original string was '0' or '00..0')
            if not stripped:
                return 0

            # Try to convert the stripped string to integer or float
            try:
                num = int(stripped)
            except ValueError:
                # If it's not an integer, try converting to float
                num = float(stripped)

        return num


    def _add_combinations_with_N(self, features):
        """
        Add combinations with 'N' to the feature dictionary if not already present.
        """

        for comb in COMBINATIONS_WITH_N:
            if comb not in features:
                features[comb] = 0
        return features

parser = argparse.ArgumentParser(description='Run the CRF classifier.')
parser.add_argument('train_data_directory', help='The train_data_directory.')
parser.add_argument('test_data_directory', help='The test_data_directory.')
parser.add_argument('output_directory',
                    help='The output directory to write to.')
args = parser.parse_args()
train_directory = args.train_data_directory
test_directory = args.test_data_directory
output_directory = args.output_directory

if not os.path.exists(os.path.join(output_directory, "pca_plots")):
    os.makedirs(os.path.join(output_directory, "pca_plots"))


# Create datasets:
train_dataset = GeneDataset(train_directory)
test_dataset = GeneDataset(test_directory)

# Dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=100048)
test_dataloader = DataLoader(test_dataset, batch_size=100048)

# Create the classifier
clf = SGDClassifier(loss="log", penalty="l2", max_iter=1000)

# Train the classifier on the training data
for X_train, y_train in train_dataloader:
    clf.partial_fit(X_train.tolist(), y_train.tolist(), classes=np.array([0, 1]))

# Evaluate the classifier on the test data
y_true, y_pred = [], []
for X_test, y_test in test_dataloader:
    y_pred_batch = clf.predict(X_test)
    y_true.extend(y_test.tolist())
    y_pred.extend(y_pred_batch.tolist())

print(classification_report(y_true, y_pred))

print(f"Classification report: {classification_report(y_true, y_pred)}")
plot_confusion_matrix(y_true, y_pred, save_path=os.path.join(
    output_directory, 'confusion_matrix.png'))
