import torch
from sklearn.metrics import classification_report
import torch.optim as optim
from torchcrf import CRF
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.nn import functional as F
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime
import os
import ast
import random
import linecache
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
from itertools import chain
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
from plotting_results import *
import json


# Make sure the device is set to cuda:"0" (first GPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
CLASS_WEIGHT = torch.tensor([1.0, 100.0]).to(device)
print(device)

class FileIterator:
    def __init__(self, directory, shuffle=True):
        self.shuffle = shuffle
        self.file_paths = [os.path.join(directory, file)
                           for file in sorted(os.listdir(directory))]
        self.file_iter = iter(self.file_generator())

    def file_generator(self):
        if self.shuffle:
            for file_path in self.file_paths:
                with open(file_path, 'r') as file:
                    line_count = sum(1 for _ in file)
                    line_indices = list(range(1, line_count + 1))
                    random.shuffle(line_indices)  # Shuffling line indices
                    for line_index in line_indices:
                        line = linecache.getline(file_path, line_index)
                        yield line
                    linecache.clearcache()
        else:
            for file_path in self.file_paths:
                with open(file_path, 'r') as file:
                    print (file_path)
                    for line in file:
                        yield line


    def __iter__(self):
        return iter(self.file_generator())

class GeneDataset(IterableDataset):
    def __init__(self, directory, max_sequence_length=4000, mode='gene', shuffle=True):
        self.shuffle = shuffle
        self.file_iterator = FileIterator(directory, shuffle= self.shuffle)
        self.max_sequence_length = max_sequence_length
        self.mode = mode

    def __iter__(self):
        for line in self.file_iterator:
            try:
                data_list = json.loads(line)
                features_list = []
                target_list = []
                mask_list = []  # List to store the mask tensors
                for data in data_list:
                    target = None
                    features = None
                    features = {"repeat": int(data.get("repetitive", None))}
                    # forgot to add "N" as base option, so adding all combinations with N if they are not present
                    #features = add_combinations_with_N(features)
                    if len(features) != 1:
                        print(f"odd number of features in token: {len(features)}")
                        # mask this value
                        features_list.append(torch.ones(129, dtype=torch.bool))
                        target_list.append(torch.tensor([0]))
                        mask_list.append(torch.ones(0, dtype=torch.bool))
                        continue

                    target = int(data.get(self.mode, None))

                    if features and target is not None:
                        features_list.append(torch.tensor(
                            [f for f in features.values()], dtype=torch.float32))
                        target_list.append(torch.tensor([target]))
                        mask_list.append(torch.ones(1, dtype=torch.bool))  # Add a mask of 1 for the sequence

                if features_list and target_list:
                    features_list = torch.stack(features_list)
                    target_list = torch.stack(target_list)
                    mask_list = torch.stack(mask_list)  # Stack the mask tensors
                    if features_list.size(0) < self.max_sequence_length:
                        pad_size = self.max_sequence_length - features_list.size(0)
                        padded_features = F.pad(features_list, (0, 0, 0, pad_size), 'constant', 0)
                        padded_target = F.pad(target_list, (0, 0, 0, pad_size), 'constant', -1)
                        padded_mask = F.pad(mask_list, (0, 0, 0, pad_size), 'constant', False)  # Pad the mask tensor
                    else:
                        padded_features = features_list
                        padded_target = target_list
                        padded_mask = mask_list

                    yield padded_features, padded_target, padded_mask  # Return the mask tensor
            except json.JSONDecodeError as e:
                print(f"Skipping line due to error: {e}")


class CRFClassifier(nn.Module):
    def __init__(self, input_dim, num_tags, hidden_dim=128):
        super(CRFClassifier, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.LSTM(input_dim, hidden_dim,
                    bidirectional=True, batch_first=True),
            # Times 2 because of bidirectionality
            nn.Linear(hidden_dim * 2, num_tags)
        )
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, x, mask):
        x, _ = self.feature_extractor[0](x)
        features = self.feature_extractor[1](x)
        mask = mask.squeeze(-1)
        return features, mask

    def decode(self, x, mask):
        x, _ = self.feature_extractor[0](x)
        features = self.feature_extractor[1](x)
        mask = mask.squeeze(-1)
        return self.crf.decode(features, mask)

    def loss(self, x, tags, mask):
        print(x)
        x, _ = self.feature_extractor[0](x)
        features = self.feature_extractor[1](x)
        print(features)
        mask = mask.squeeze(-1)
        adjusted_features = features + CLASS_WEIGHT.view(1, 1, -1)
        return -self.crf(adjusted_features, tags, mask)


def parse_number(s):
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

def get_model_predictions_and_labels(model, dataloader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels, mask in dataloader:  # Include the mask tensor
            inputs, labels, mask = inputs.to(device), labels.to(device), mask.to(device)
            outputs = model.decode(inputs, mask)  # Pass the mask tensor to the decode method
            predicted_labels = outputs
            labels = labels[mask]
            y_true.extend(labels.tolist())
            y_pred.extend(predicted_labels)
    y_true = flatten_list(y_true)
    y_pred = flatten_list(y_pred)
    return y_true, y_pred

def train_crf_classifier(train_dataloader, input_dim, num_tags, num_epochs):
    model = CRFClassifier(input_dim, num_tags).to(device)
    learning_rate = 0.1  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()
    batch_losses = []
    # Create a list to store the loss at each epoch
    epoch_losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (inputs, tags, mask) in enumerate(train_dataloader):
            inputs, tags, mask = inputs.to(device), tags.to(device), mask.to(device)

            optimizer.zero_grad()
            with autocast():
                tags = tags.squeeze(-1)
                print_nan_details(inputs, 'Inputs')
                print_nan_details(tags, 'Tags')
                print_nan_details(mask, 'Mask')
                loss = model.loss(inputs, tags, mask)  # Pass the mask tensor to the loss function
            print(loss.shape)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            batch_losses.append(loss.item())
            print(loss.item())
        # Compute average loss for this epoch
        avg_loss = total_loss / (i+1)
        print(f"Epoch {epoch+1}: Loss = {avg_loss}")
        epoch_losses.append(avg_loss)

    return model, epoch_losses, batch_losses  # return losses along with the model

def flatten_list(nested_list):
    flattened_list = []
    for item in nested_list:
        if isinstance(item, list):
            flattened_list.extend(flatten_list(item))
        else:
            flattened_list.append(item)
    return flattened_list

def print_nan_details(tensor, tensor_name):
    if torch.isnan(tensor).any():
        print(f'{tensor_name} contains NaN values')

parser = argparse.ArgumentParser(description='Run the CRF classifier.')
parser.add_argument('train_data_directory', help='The train_data_directory.')
parser.add_argument('test_data_directory', help='The test_data_directory.')
parser.add_argument('output_directory', help='The output directory to write to.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to use.')
parser.add_argument('--mode', type=str, default="exon",
                    help='Target to use for prediction, either gene or exon.')

args = parser.parse_args()
train_directory = args.train_data_directory
test_directory = args.test_data_directory
output_directory = args.output_directory
mode = args.mode

# Create datasets:
train_dataset = GeneDataset(train_directory, mode=mode)
test_dataset = GeneDataset(test_directory, mode=mode)

# Dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=64)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Train the model and get the losses for each epoch
model, epoch_losses, batch_losses = train_crf_classifier(train_dataloader, input_dim=1, num_tags=2, num_epochs=args.epochs)

# Save the model after training
torch.save(model.state_dict(), os.path.join(
    output_directory, f'crf_{timestamp}.pth'))

# Plot the training loss curve
plot_loss_curve(epoch_losses, save_path=os.path.join(
    output_directory, f'training_loss_curve_{timestamp}.png'))

# Plot the training batch loss curve
plot_batch_losses(batch_losses, save_path=os.path.join(
    output_directory, f'training_batch_loss_curve_{timestamp}.png'))

#Plot results
y_true, y_pred = get_model_predictions_and_labels(model, test_dataloader)
print(f"Classification report: {classification_report(y_true, y_pred)}")
plot_confusion_matrix(y_true, y_pred, save_path=os.path.join(output_directory, f'confusion_matrix_{timestamp}.png'))
plot_sequence_labels(y_true, y_pred, save_path=os.path.join(
    output_directory, f'linear_prediction_path_{timestamp}.png'))
