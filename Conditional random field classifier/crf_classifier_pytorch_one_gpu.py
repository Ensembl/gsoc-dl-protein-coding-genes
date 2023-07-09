import torch
import torch.optim as optim
from torchcrf import CRF
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.nn import functional as F
from torch import nn
from torch.cuda.amp import autocast, GradScaler
import os
import ast
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
from itertools import chain
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
from plotting_results import *

# Make sure the device is set to cuda:"0" (first GPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class FileIterator:
    def __init__(self, directory):
        self.file_paths = [os.path.join(directory, file) for file in sorted(os.listdir(directory))]
        self.file_iter = iter(self.file_generator())

    def file_generator(self):
        for file_path in self.file_paths:
            with open(file_path, 'r') as file:
                for line in file:
                    yield line

    def __iter__(self):
        return self.file_iter

class GeneDataset(IterableDataset):
    def __init__(self, directory, max_sequence_length=4000):
        self.file_iterator = FileIterator(directory)
        self.max_sequence_length = max_sequence_length

    def __iter__(self):
        for line in self.file_iterator:
            try:
                data_list = json.loads(line)
                features_list = []
                target_list = []
                for data in data_list:
                    features = {k: parse_number(v) for k, v in data.items() if k != 'gene'}
                    target = parse_number(data.get('gene', None))
                    if features and target:
                        features_list.append(torch.tensor([f for f in features.values()]))
                        target_list.append(torch.tensor(target))
                if features_list and target_list:
                    # Pad or truncate sequences to the desired length
                    padded_features = pad_sequence(features_list, batch_first=True).squeeze()
                    padded_target = pad_sequence(target_list, batch_first=True).squeeze()

                    if padded_features.size(0) > self.max_sequence_length:
                        padded_features = padded_features[:self.max_sequence_length]  # Truncate features
                    if padded_target.size(0) > self.max_sequence_length:
                        padded_target = padded_target[:self.max_sequence_length]  # Truncate target

                    yield padded_features, padded_target
            except json.JSONDecodeError as e:
                print(f"Skipping line due to error: {e}")

class CRFClassifier(nn.Module):
    def __init__(self, input_dim, num_tags):
        super(CRFClassifier, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 68),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_tags)
        )
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.crf.decode(features)
        return output

    def loss(self, x, tags):
        features = self.feature_extractor(x)
        return -self.crf(features, tags)

def parse_number(s):
    # If the string starts with '0.', don't strip the leading zero
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
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(outputs)
    return all_labels, all_predictions

def train_crf_classifier(train_dataloader, input_dim, num_tags, num_epochs):
    model = CRFClassifier(input_dim, num_tags).to(device)

    optimizer = optim.Adam(model.parameters())
    scaler = GradScaler()

    # Create a list to store the loss at each epoch
    epoch_losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (inputs, tags) in enumerate(train_dataloader):
            inputs, tags = inputs.to(device), tags.to(device)
            optimizer.zero_grad()
            with autocast():
                loss = model.loss(inputs, tags)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        # Compute average loss for this epoch
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss}")
        epoch_losses.append(avg_loss)

    return model, epoch_losses  # return losses along with the model

def plot_loss_curve(epoch_losses, save_path=None):
    plt.plot(epoch_losses)
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if save_path:
        plt.savefig(save_path, dpi=900)
    plt.show()

parser = argparse.ArgumentParser(description='Run the CRF classifier.')
parser.add_argument('train_data_directory', help='The train_data_directory.')
parser.add_argument('test_data_directory', help='The test_data_directory.')
parser.add_argument('output_directory', help='The output directory to write to.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to use.')
args = parser.parse_args()
train_directory = args.train_data_directory
test_directory = args.test_data_directory
output_directory = args.output_directory

# Now, to create your datasets:
train_dataset = GeneDataset(train_directory)
test_dataset = GeneDataset(test_directory)

# Dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=32)
test_dataloader = DataLoader(test_dataset, batch_size=32)

# Train the model and get the losses for each epoch
model, epoch_losses = train_crf_classifier(train_dataloader, input_dim=128, num_tags=2, num_epochs=args.epochs)

# Save the model after training
torch.save(model.state_dict(), os.path.join(output_directory, 'crf.pth'))

# Plot the training loss curve
plot_loss_curve(epoch_losses, save_path=os.path.join(output_directory, 'training_loss_curve.png'))

# Assuming you have trained your model and have your dataloader ready:
y_true, y_pred = get_model_predictions_and_labels(model, test_dataloader)

# Then you can use the data to plot the graphs:
plot_confusion_matrix(y_true, y_pred, save_path=os.path.join(output_directory, 'confusion_matrix.png'))
