import random
import linecache
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
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt

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
                line_count = sum(1 for _ in file)
                line_indices = list(range(1, line_count + 1))
                random.shuffle(line_indices)  # Shuffling line indices
                for line_index in line_indices:
                    line = linecache.getline(file_path, line_index)
                    yield line
                linecache.clearcache()

    def __iter__(self):
        return iter(self.file_generator())

class GeneDataset(IterableDataset):
    def __init__(self, directory, max_sequence_length=4000):
        self.file_iterator = FileIterator(directory)
        self.max_sequence_length = max_sequence_length
        


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
                    features = {k: self._parse_number(v)/250 for k, v in data.items() if k != 'gene'}
                    if features["strand"] != -1:
                        if -1 in features.values():
                            print(f"Weird features: {features}")
                            continue
                    # forgot to add "N" as base option, so adding all combinations with N if they are not present
                    #features = self._add_combinations_with_N(features)
                    if len(features) != 129:
                        print(f"odd number of features in token: {features}")
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
                print (f"Genes in sequence: {genes_in_sequence}")
                yield features_list, target_list  # Return the mask tensor
            except json.JSONDecodeError as e:
                print(f"Skipping line due to error: {e}")

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

def transform_and_plot(features, targets, batch_count):
        pca = PCA(n_components=2)
        try:
            transformed_features = pca.fit_transform(features)
            plt.figure(figsize=(8, 6))
            plt.scatter(transformed_features[:, 0],
                        transformed_features[:, 1], c=targets)
            plt.colorbar()
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.title(f'PCA of data for batch {batch_count}')
            plt.savefig(os.path.join(output_directory, "pca_plots",
                        f"pca_batch_{batch_count}.png"))
            plt.close()

            return transformed_features
        except Exception as e:
            print("An error occurred: ", str(e))
            print("Features at the time of error: ", features)

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
train_dataloader = DataLoader(train_dataset, batch_size=1048)
test_dataloader = DataLoader(test_dataset, batch_size=1048)

# Initialize lists to store results
weights = []
precisions = []
recalls = []

# Test out different weights
for i in range(0, 6):
    weight = {0: 1, 1: 10 ** i}  # Increasing weight for the second class
    weights.append(weight[1])

    # Create the classifier
    clf = SGDClassifier(loss="modified_huber", penalty="l2",
                        max_iter=1000, class_weight=weight)

    # Train the classifier on the training data
    batch_count = 0
    for X_train, y_train in train_dataloader:
        X_train = [[element.tolist() for element in inner_list]
                   for inner_list in X_train]
        y_train = [[element.tolist() for element in inner_list]
                   for inner_list in y_train]
        # Convert y_train to a numpy array and flatten it
        y_train = np.array(y_train)
        y_train = y_train.flatten()
        X_train = np.array(X_train)
        X_train = np.reshape(X_train, (-1, X_train.shape[1]))
        transform_and_plot(X_train, y_train, batch_count)
        clf.partial_fit(X_train, y_train, classes=np.array([0, 1]))
        batch_count += 1

    # Evaluate the classifier on the test data
    y_true, y_pred = [], []
    for X_test, y_test in test_dataloader:
        X_test = [[element.tolist() for element in inner_list]
                  for inner_list in X_test]
        y_test = [[element.tolist() for element in inner_list]
                  for inner_list in y_test]
        # Convert y_train to a numpy array and flatten it
        y_test = np.array(y_test)
        y_test = y_test.flatten()
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (-1, X_test.shape[1]))
        y_pred_batch = clf.predict(X_test)
        if len(y_true) == len(y_pred):
            y_true.extend(y_test.tolist())
            y_pred.extend(y_pred_batch.tolist())

    # Calculate and store precision and recall
    precision = precision_score(y_true, y_pred, pos_label=1)
    recall = recall_score(y_true, y_pred, pos_label=1)
    precisions.append(precision)
    recalls.append(recall)
    try:
        print(
            f"Classification report {weight}: {classification_report(y_true, y_pred)}")
    except:
        print("Error in Classification report")
    plot_confusion_matrix(y_true, y_pred, save_path=os.path.join(
        output_directory, f'{weight[1]}_confusion_matrix.png'))

# Plot precision and recall for different weights
plt.figure(figsize=(12, 6))

plt.plot(weights, precisions, marker='o',
         linestyle='-', color='b', label='Precision')
plt.plot(weights, recalls, marker='o',
         linestyle='-', color='r', label='Recall')

plt.xlabel('Weight for second class')
plt.ylabel('Score')
plt.title('Effect of Class Weight on Precision and Recall')
plt.legend()
plt.savefig(os.path.join(
    output_directory, f'precision_recall_per_weight.png'), dpi=900)
