import torch
import torch.optim as optim
from torchcrf import CRF
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
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

# Make sure the device is set to cuda:0 (first GPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
class GeneDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.LongTensor(self.labels[idx])

class CRFClassifier(nn.Module):
    def __init__(self, input_dim, num_tags):
        super(CRFClassifier, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
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
# Initialize the distributed environment
dist.init_process_group(backend='nccl')

# Get the rank (ID of the current process) and the world size (total number of processes)
rank = dist.get_rank()
world_size = dist.get_world_size()

# Divide the files between the processes
train_directory = args.train_data_directory
test_directory = args.test_data_directory
output_directory = args.output_directory
train_files = sorted(os.listdir(train_directory))
test_files = sorted(os.listdir(test_directory))
train_files = train_files[rank::world_size]
test_files = test_files[rank::world_size]

# Now, load and process the files assigned to this process
train_data = []
train_labels = []
for file in train_files:
    with open(os.path.join(train_directory, file), 'r') as f:
        data = ast.literal_eval(f.read())
        train_data.append(data["features"])
        train_labels.append(data["labels"])

test_data = []
test_labels = []
for file in test_files:
    with open(os.path.join(test_directory, file), 'r') as f:
        data = ast.literal_eval(f.read())
        test_data.append(data["features"])
        test_labels.append(data["labels"])

# Create dataloaders
train_dataset = GeneDataset(train_data, train_labels)
test_dataset = GeneDataset(test_data, test_labels)
train_sampler = DistributedSampler(train_dataset)
test_sampler = DistributedSampler(test_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=32)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=32)

# Train the model and get the losses for each epoch
model, epoch_losses = train_crf_classifier(train_dataloader, input_dim=128, num_tags=2, num_epochs=100)

# save the model after training
if rank == 0:  # only save on the main process
    torch.save(model.state_dict(), os.path.join(output_directory,'crf.pth'))
    
# Plot the training loss curve
plot_loss_curve(epoch_losses, save_path= os.path.join(output_directory,'training_loss_curve.png'))

# Assuming you have trained your model and have your dataloader ready:
y_true, y_pred = get_model_predictions_and_labels(model, test_dataloader)

# Then you can use the data to plot the graphs:
plot_confusion_matrix(y_true, y_pred, save_path=os.path.join(output_directory,'confusion_matrix.png'))

