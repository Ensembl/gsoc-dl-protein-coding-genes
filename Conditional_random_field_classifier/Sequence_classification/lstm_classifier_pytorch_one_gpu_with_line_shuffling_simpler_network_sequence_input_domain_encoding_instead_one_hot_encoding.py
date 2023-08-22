import torch.nn as nn
import os
import torch
import random
import sys
import json
import linecache
import argparse
import pandas as pd
from datetime import datetime
import numpy as np
import torch.optim as optim
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import  DataLoader, IterableDataset
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.nn.init as init
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from plotting_results import *
import torch.autograd.profiler as profiler


# Make sure the device is set to cuda:"0" (first GPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

# Things for 
# Original one-hot encoded keys as tensors
# Corresponding 4D values for DNA representation
VALUES_ENCODING = [
    torch.tensor([1, 0, 0, 0]),
    torch.tensor([0, 1, 0, 0]),
    torch.tensor([0, 0, 1, 0]),
    torch.tensor([0, 0, 0, 1]),
    torch.tensor([0.25, 0.25, 0.25, 0.25]),
    torch.tensor([0.5, 0, 0.5, 0]),
    torch.tensor([0, 0.5, 0, 0.5]),
    torch.tensor([0, 0, 0.5, 0.5]),
    torch.tensor([0.5, 0.5, 0, 0]),
    torch.tensor([0, 0.5, 0.5, 0]),
    torch.tensor([0.5, 0, 0, 0.5]),
    torch.tensor([1/3, 1/3, 0, 1/3]),
    torch.tensor([0, 1/3, 1/3, 1/3]),
    torch.tensor([1/3, 0, 1/3, 1/3]),
    torch.tensor([1/3, 1/3, 1/3, 0])
]

# Create a conversion tensor of shape (15, 4) for fast lookup
CONVERSION_TENSOR = torch.stack(VALUES_ENCODING)


class FileIterator:
    def __init__(self, directory, shuffle=True):
        self.shuffle = shuffle
        self.file_paths = [os.path.join(directory, file)
                        for file in sorted(os.listdir(directory)) if "reverse" in file or "forward" in file]
        self.file_iter = iter(self.file_generator())

    def file_generator(self):
        if self.shuffle:
            for file_path in self.file_paths:
                current_file = os.path.splitext(
                    os.path.basename(file_path))[0]
                with open(file_path, 'r') as file:
                    line_count = sum(1 for _ in file)
                    line_indices = list(range(1, line_count + 1))
                    random.shuffle(line_indices)  # Shuffling line indices
                    for line_index in line_indices:
                        line = linecache.getline(file_path, line_index)
                        yield line, current_file
                    linecache.clearcache()
        else:
            for file_path in self.file_paths:
                with open(file_path, 'r') as file:
                    current_file = os.path.splitext(
                        os.path.basename(file_path))[0]
                    for line in file:
                        yield line, current_file


    def __iter__(self):
        return iter(self.file_generator())

class GeneDataset(IterableDataset):
    def __init__(self, directory, max_sequence_length=4000, mode='gene', shuffle=True):
        self.shuffle = shuffle
        self.file_iterator = FileIterator(directory, shuffle= self.shuffle)
        self.max_sequence_length = max_sequence_length
        self.mode = mode

    def __iter__(self):
        for line, current_file in self.file_iterator:
            try:
                data_list = json.loads(line)
                features_sequence_list = []
                features_general_list = []
                target_list = []
                mask_list = []  # List to store the mask tensors
                positions_list = []
                exons = 0
                for data in data_list:
                    target = None

                    # The sequence is encoded, so the sequence features have a different dimensionality than the general features
                    features_general = None
                    features_sequence = None
                    # convert_one_hot_to_4d needs to be done to try to reduce dimensionality of input embedding
                    features_sequence = convert_sequence_to_4D(torch.tensor(
                        (data["sequence"]), dtype=torch.float32))
                    #print(features_sequence)
                    features_general = torch.tensor([v if k not in ('repetitive') else (1-v)*10 for k, v in data.items(
                    ) if k != 'gene' and k != 'exon' and k != 'position' and k != 'sequence'], dtype=torch.float32)
                    position = int(data.get("position", None))
                    target = int(data.get(self.mode, None))
                    
                    if target == 1:
                        exons +=1

                    if features_sequence is not None and features_general is not None and target is not None:
                        features_general_list.append(features_general)
                        features_sequence_list.append(features_sequence)
                        target_list.append(torch.tensor([target]))
                        mask_list.append(torch.ones(1, dtype=torch.bool))  # Add a mask of 1 for the sequence
                        positions_list.append(torch.tensor([position]))
                features_general_list = torch.stack(features_general_list)
                features_sequence_list = torch.stack(features_sequence_list)
                
                target_list = torch.stack(target_list)
                mask_list = torch.stack(mask_list) 
                positions_list = torch.stack(positions_list)
                # Pad if features sequences are less than 4000 tokens long
                if features_sequence_list.size(0) < self.max_sequence_length:
                    pad_size = self.max_sequence_length - \
                        features_sequence_list.size(0)
                    
                    padded_features_sequence = F.pad(
                        features_sequence_list, (0, 0, 0, 0, 0, pad_size), 'constant', 0)
                    padded_features_general = F.pad(
                        features_general_list, (0, 0, 0, pad_size), 'constant', 0)
                    padded_target = F.pad(target_list, (0, 0, 0, pad_size), 'constant', -1)
                    padded_mask = F.pad(mask_list, (0, 0, 0, pad_size), 'constant', False)  # Pad the mask tensor
                    padded_positions = F.pad(
                        positions_list, (0, 0, 0, pad_size), 'constant', -1)
                else:
                    padded_features_sequence = features_sequence_list
                    padded_features_general = features_general_list
                    padded_target = target_list
                    padded_mask = mask_list
                    padded_positions = positions_list
                # Return the mask tensor
                yield padded_features_sequence, padded_features_general, padded_target, padded_mask, padded_positions, current_file
            except json.JSONDecodeError as e:
                print(f"Skipping line due to error: {e}")


class LSTMClassifier(nn.Module):
    def __init__(self, seq_input_dim, normal_input_dim, hidden_dim=64, num_layers=2, dropout_prob=0.1):
        super(LSTMClassifier, self).__init__()

        # For the sequence data
        self.lstm = nn.LSTM(seq_input_dim, hidden_dim, num_layers=num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)

        # Intermediate fully connected layer
        # The input will be the concatenated outputs of LSTM and normal features
        self.fc1 = nn.Linear(hidden_dim*2 + normal_input_dim, hidden_dim)

        # Dropout layer
        self.dropout_fc1 = nn.Dropout(p=dropout_prob)

        # Final fully connected layer
        self.fc2 = nn.Linear(hidden_dim, 1)

        # Apply the initialization
        init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')

    def forward(self, seq_data, normal_data):
        # Process sequence data through LSTM
        lstm_out, _ = self.lstm(seq_data)

        # Optionally, get the last timestep's output for each sequence
        lstm_out = lstm_out[:, -1, :]
        print(seq_data.shape)
        print(normal_data.shape)
        print(lstm_out.shape)

        # Concatenate LSTM output and normal features
        combined_data = torch.cat((lstm_out, normal_data), dim=1)

        # Pass combined data through fully connected layers
        x = self.fc1(combined_data)
        x = self.dropout_fc1(x)
        x = self.fc2(x)

        # Apply sigmoid activation to get the probability
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
    print(f"Precision:{precision}, Recall: {recall}")
    # Calculating F-beta score
    f_beta = (1 + beta**2) * (precision * recall) / \
        (beta**2 * precision + recall + 1e-5)
    return 1 - f_beta.mean(), precision, recall

def train_lstm_classifier(train_dataloader, input_dim, num_tags, num_epochs):
    model = LSTMClassifier(seq_input_dim=4, normal_input_dim=3).to(device)
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Define the scheduler
    # Update every 10 epochs with decay factor 0.1
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    # The beta value defines the ration of precision and recall of the underrepresented class to be used (smaller for higher precision)
    beta_value = 2
    batch_losses = []
    epoch_losses = []
    exons_per_batch = []
    writer = SummaryWriter()
    clip_value = 1.0
    for epoch in range(num_epochs):
        total_loss = 0
        skipped_batches = 0
        for i, (seq_data, normal_data, tags, mask, _, _) in enumerate(train_dataloader):
            print(
                f"Memory before data loading: {torch.cuda.memory_allocated() / (2**20):.2f} MB")
            seq_data, normal_data, tags, mask = filter_inputs_with_threshold_targets(
                seq_data, normal_data, tags.squeeze(-1), mask.squeeze(-1))
            if torch.sum(tags == 1).item() == 0:
                skipped_batches += 1
                print("continued")
                continue
            seq_data = seq_data.view(-1, seq_data.size(2), seq_data.size(3))
            normal_data = normal_data.view(-1, normal_data.size(2))
            tags = tags.view(-1, 1)
            mask = mask.view(-1, 1)
            seq_data, normal_data, tags = seq_data.to(device), normal_data.to(device), tags.to(device)
            mask = mask.to(device)
            print(f"Size of seq_data: {tensor_size_in_MB(seq_data):.2f} MB")
            print(f"Size of normal_data: {tensor_size_in_MB(normal_data):.2f} MB")

            torch.cuda.empty_cache()
            optimizer.zero_grad()
            outputs = model(seq_data, normal_data)
            print(f"Outputs:{outputs.shape}, tags: {tags.shape}, maks: {mask.shape}")
            print(
                f"Memory after forwardpass: {torch.cuda.memory_allocated() / (2**20):.2f} MB")
            print(
                f"Cached memory after forwardpass: {torch.cuda.memory_cached() / (2**20):.2f} MB")

            loss, precision, recall = f_beta_loss(
                outputs[mask], tags[mask], beta=beta_value)
            loss.backward()
            print(
                f"Memory after backwardpass: {torch.cuda.memory_allocated() / (2**20):.2f} MB")
            print(
                f"Cached memory after backwardpass: {torch.cuda.memory_cached() / (2**20):.2f} MB")

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            writer.add_scalar('loss', loss, i)
            writer.add_scalar('precision', precision, i)
            writer.add_scalar('recall', recall, i)
            optimizer.step()
            total_loss += loss.item()
            batch_losses.append(loss.item())
            print(loss.item())
            print(f'Exons: {torch.sum(tags == 1).item()}')
            exons_per_batch.append(torch.sum(tags == 1).item())
        if i+1 - skipped_batches == 0:
            continue
        avg_loss = total_loss / (i+1 - skipped_batches)
        print(f"Epoch {epoch+1}: Loss = {avg_loss}")
        epoch_losses.append(avg_loss)
        scheduler.step()
    writer.close()
    return model, epoch_losses, batch_losses  # return losses along with the model


def filter_inputs_with_threshold_targets(seq_data, normal_data, targets, mask, threshold=150):
    # Count the number of occurrences of '1' in each row of the targets
    rows_with_more_than_threshold_ones = (targets == 1).sum(dim=1) >= threshold

    # Use boolean indexing to keep only the rows that have more than the given threshold occurrences of 1
    filtered_seq_data = seq_data[rows_with_more_than_threshold_ones]
    filtered_normal_data = normal_data[rows_with_more_than_threshold_ones]
    filtered_targets = targets[rows_with_more_than_threshold_ones]
    filtered_mask = mask[rows_with_more_than_threshold_ones]

    return filtered_seq_data, filtered_normal_data, filtered_targets, filtered_mask


def convert_sequence_to_4D(input_sequence):
    """
    Convert a sequence of one-hot encoded embeddings to a sequence of 4D tensors.

    Args:
    - input_sequence (Tensor): A 2D tensor where each row is a one-hot encoded embedding.

    Returns:
    - Tensor: A 2D tensor where each row is the corresponding 4D tensor.
    """
    output_sequence = []
    for i in range(input_sequence.size(0)):
        idx = (input_sequence[i] == 1).nonzero(as_tuple=True)[0].item()
        output_sequence.append(CONVERSION_TENSOR[idx])

    return torch.stack(output_sequence)

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

def get_model_predictions_and_labels(model, dataloader, threshold=0.5):
    model.eval()
    y_true = []
    y_pred = []
    rows = []
    torch.cuda.empty_cache()
    with torch.no_grad():
        for seq_data, normal_data, labels, mask, positions, sequence_name in dataloader:
            print("Still living")
            seq_data, normal_data, labels, positions = seq_data.to(
                device), normal_data.to(device), labels.to(device), positions.to(device)
            positions = positions.squeeze()
            seq_data = seq_data.view(-1, seq_data.size(2), seq_data.size(3))
            normal_data = normal_data.view(-1, normal_data.size(2))
            labels = labels.view(-1, 1)
            mask = mask.view(-1, 1)
            
            mask = mask.squeeze(-1).to(device).flatten()
            print(f"Size of seq_data: {tensor_size_in_MB(seq_data):.2f} MB")
            print(f"Size of normal_data: {tensor_size_in_MB(normal_data):.2f} MB")
            with profiler.profile(use_cuda=True) as prof:
                with profiler.record_function("model_inference"):
                    outputs = model(seq_data, normal_data).flatten()
            print(prof.key_averages().table(sort_by="cuda_time_total"))
            labels =labels.flatten()
            predicted_labels = (outputs > threshold).float()
            y_true.extend(labels[mask].tolist())
            y_pred.extend(predicted_labels[mask].tolist())
            valid_positions = positions[mask].tolist()
            valid_predicted_scores = outputs[mask].tolist()
            for  pos, score in zip( valid_positions, valid_predicted_scores):
                rows.append((sequence_name, pos, score))
            torch.cuda.empty_cache()

    df = pd.DataFrame(
            rows, columns=['filename', 'position', 'predicted_score'])

    return y_true, y_pred, outputs, df

def tensor_size_in_MB(tensor):
    return tensor.element_size() * tensor.nelement() / (1024 * 1024)

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

# Create Output directory
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    print(f"Directory {output_directory} created.")
else:
    print(f"Directory {output_directory} already exists.")

if not os.path.exists(os.path.join(output_directory, "pca_plots")):
    os.makedirs(os.path.join(output_directory, "pca_plots"))
    print(f"Directory {os.path.join(output_directory, 'pca_plots')} created.")
else:
    print(f"Directory {os.path.join(output_directory, 'pca_plots')} already exists.")

# Create datasets:
train_dataset = GeneDataset(train_directory, mode=mode, shuffle=True)
test_dataset = GeneDataset(test_directory, mode=mode, shuffle=True)

# Dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=50)
test_dataloader = DataLoader(test_dataset, batch_size=1)

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
y_true, y_pred, y_probabilities, df = get_model_predictions_and_labels(model, test_dataloader)
print(f"Classification report: {classification_report(y_true, y_pred)}")
# Save the DataFrame to a CSV file
df.to_csv(os.path.join(
    output_directory, f'predictions_{timestamp}.csv'), index=False)

plot_confusion_matrix(y_true, y_pred, save_path=os.path.join(output_directory, f'confusion_matrix_{timestamp}.png'))
plot_sequence_labels(y_true[:512], y_probabilities[:512], save_path=os.path.join(
    output_directory, f'linear_prediction_path_{timestamp}.png'))
