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
from torch.utils.tensorboard import SummaryWriter
import random
import linecache
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
from itertools import chain
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
from plotting_results import *
import torch.nn.init as init
from sklearn.decomposition import PCA
from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn.functional as F
import json

# Make sure the device is set to cuda:"0" (first GPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
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
                print(file_path)
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
                exons = 0
                for data in data_list:
                    target = None
                    features = None
                    features = {k: parse_number(
                        v)/25 if k not in ('repetitive') else parse_number(1-v)*100 for k, v in data.items() if k != 'gene' and k != 'exon' and k != 'position'}
                    # forgot to add "N" as base option, so adding all combinations with N if they are not present
                    #features = add_combinations_with_N(features)
                    if len(features) != 128:
                        print(f"odd number of features in token: {len(features)}")
                        # mask this value
                        features_list.append(torch.ones(128, dtype=torch.bool))
                        target_list.append(torch.tensor([0]))
                        mask_list.append(torch.ones(0, dtype=torch.bool))
                        continue

                    target = int(data.get(self.mode, None))
                    if target == 1:
                        exons +=1

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
                    #print(exons)
                    yield padded_features, padded_target, padded_mask  # Return the mask tensor
            except json.JSONDecodeError as e:
                print(f"Skipping line due to error: {e}")


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, num_tags, hidden_dim=128, num_layers=2, dropout_prob=0.1):
        super(LSTMClassifier, self).__init__()

        # Define a multi-layer, bidirectional LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)

        # Define a fully connected layer to map the LSTM output to an intermediate space
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)

        # Add a Dropout layer after the intermediate fully connected layer
        self.dropout_fc1 = nn.Dropout(p=dropout_prob)

        # Define the final fully connected layer that maps to the desired number of tags
        self.fc2 = nn.Linear(hidden_dim, 1)

        # Apply the initialization here
        init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')

    def forward(self, x):
        # Apply the LSTM to the input
        x, _ = self.lstm(x)

        # Apply the fully connected layer to the LSTM output
        x = self.fc1(x)

        # Apply dropout to the output of the first fully connected layer
        x = self.dropout_fc1(x)

        # Apply the final fully connected layer
        x = self.fc2(x)

        # Apply a sigmoid activation to get the probability
        x = torch.sigmoid(x)

        return x


def f_beta_loss(y_pred, y_true, beta=1):
    # Calculating Precision and Recall
    
    y_pred = y_pred.squeeze()
    tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

    precision = tp / (tp + fp + 1e-5)
    recall = tp / (tp + fn + 1e-5)
    print(precision, recall)
    # Calculating F-beta score
    f_beta = (1 + beta**2) * (precision * recall) / \
        (beta**2 * precision + recall + 1e-5)
    return 1 - f_beta.mean()

def train_lstm_classifier(train_dataloader, input_dim, num_tags, num_epochs):
    model = LSTMClassifier(input_dim, num_tags).to(device)
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #weights = torch.tensor([1.0, 25.0], dtype=torch.float32).to(device)
    #loss_function = nn.CrossEntropyLoss(weight=weights)

    # Define the scheduler
    # Update every 10 epochs with decay factor 0.1
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    beta_value = 0.5
    batch_losses = []
    epoch_losses = []
    exons_per_batch = []
    writer = SummaryWriter()

    for epoch in range(num_epochs):
        total_loss = 0
        for i, (inputs, tags, mask) in enumerate(train_dataloader):

            inputs, tags, mask = filter_inputs_with_threshold_targets(
                inputs, tags.squeeze(-1), mask.squeeze(-1))
            if torch.sum(tags == 1).item() == 0:
                continue
            if epoch == 0:
                inputs_numpy = inputs.cpu().numpy()  # Convert the inputs tensor to a NumPy array
                tags_numpy = tags.cpu().numpy()     # Convert the tags tensor to a NumPy array

                # Reshape the arrays, keeping the last dimension and merging all others
                inputs_reshaped = np.reshape(inputs_numpy, (-1, inputs_numpy.shape[-1]))
                tags_flatten = tags_numpy.flatten()

                transform_and_plot(inputs_reshaped, tags_flatten, i)
            inputs, tags = inputs.to(device), tags.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = f_beta_loss(outputs[mask], tags[mask], beta=beta_value)
            loss.backward()

            writer.add_scalar('loss', loss, epoch)

            optimizer.step()
            total_loss += loss.item()
            batch_losses.append(loss.item())
            print(loss.item())
            print(f'Exons: {torch.sum(tags == 1).item()}')
            exons_per_batch.append(torch.sum(tags == 1).item())
        avg_loss = total_loss / (i+1)
        print(f"Epoch {epoch+1}: Loss = {avg_loss}")
        epoch_losses.append(avg_loss)
        scheduler.step()
    writer.close()
    return model, epoch_losses, batch_losses  # return losses along with the model


def filter_inputs_with_threshold_targets(inputs, targets, mask, threshold=0):
    # Count the number of occurrences of '1' in each row of the targets
    rows_with_more_than_threshold_ones = (targets == 1).sum(dim=1) > threshold

    # Use boolean indexing to keep only the rows that have more than the given threshold occurrences of 1
    filtered_inputs = inputs[rows_with_more_than_threshold_ones]
    filtered_targets = targets[rows_with_more_than_threshold_ones]
    filtered_mask = mask[rows_with_more_than_threshold_ones]

    return filtered_inputs, filtered_targets, filtered_mask


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
                        f"pca_batch_{batch_count}_{timestamp}.png"))
            plt.close()

            return transformed_features
        except Exception as e:
            print("An error occurred: ", str(e))
            print("Features at the time of error: ", features)


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


def get_model_predictions_and_labels(model, dataloader, threshold=0.4):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels, mask in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            mask = mask.squeeze(-1).to(device).flatten()
            outputs = model(inputs).flatten()
            labels =labels.flatten()
            print(outputs)
            y_probability = outputs
            predicted_labels = (outputs > threshold).float()
            print(predicted_labels)
            print(mask)
            y_true.extend(labels[mask].tolist())
            y_pred.extend(predicted_labels[mask].tolist())

    return y_true, y_pred, y_probability

def flatten_list(nested_list):
    flattened_list = []
    for item in nested_list:
        if isinstance(item, list):
            flattened_list.extend(flatten_list(item))
        else:
            flattened_list.append(item)
    return flattened_list

def calculate_accuracy(y_true, y_pred):
    correct = 0
    total = 0
    for true, pred in zip(y_true, y_pred):
        correct += int(true == pred)
        total += 1
    return correct / total

def print_nan_details(tensor, tensor_name):
    if torch.isnan(tensor).any():
        print(f'{tensor_name} contains NaN values')

parser = argparse.ArgumentParser(description='Run the lstm classifier.')
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
train_dataset = GeneDataset(train_directory, mode=mode, shuffle=True)
test_dataset = GeneDataset(test_directory, mode=mode, shuffle=False)

# Dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=64)
test_dataloader = DataLoader(test_dataset, batch_size=64)

# Train the model and get the losses for each epoch
model, epoch_losses, batch_losses = train_lstm_classifier(
    train_dataloader, input_dim=128, num_tags=2, num_epochs=args.epochs)

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
y_true, y_pred, y_probability = get_model_predictions_and_labels(model, train_dataloader)
print(f"Classification report: {classification_report(y_true, y_pred)}")
plot_confusion_matrix(y_true, y_pred, save_path=os.path.join(output_directory, f'confusion_matrix_{timestamp}.png'))
plot_sequence_labels(y_true, y_probability, save_path=os.path.join(
    output_directory, f'linear_prediction_path_{timestamp}.png'))
