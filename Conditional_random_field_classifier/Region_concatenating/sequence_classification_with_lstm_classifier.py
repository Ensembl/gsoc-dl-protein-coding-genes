import argparse
from Bio import SeqIO
from data_preprocession_with_size_redistribution_sequence_input import process_fasta
import pandas as pd
import json
import torch
from torch.autograd import Variable
import os
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, FeatureLocation
from Bio import SeqIO
import torch.nn as nn
import torch.nn.init as init
import linecache
import random
from torch.utils.data import DataLoader, IterableDataset
from torch.nn import functional as F
from datetime import datetime
import torch.autograd.profiler as profiler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

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

class FileIterator:
    def __init__(self, file_paths, shuffle=False):
        self.shuffle = shuffle
        self.file_paths = file_paths
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
    def __init__(self, file_paths, max_sequence_length=4000, mode='gene', shuffle=False):
        self.shuffle = shuffle
        self.file_iterator = FileIterator(file_paths, shuffle=self.shuffle)
        self.max_sequence_length = max_sequence_length
        self.mode = mode

    def __iter__(self):
        for line, current_file in self.file_iterator:
            try:
                data_list = json.loads(line)
                features_sequence_list = []
                features_general_list = []
                mask_list = []  # List to store the mask tensors
                positions_list = []

                for data in data_list:
                    # The sequence is encoded, so the sequence features have a different dimensionality than the general features
                    features_general = None
                    features_sequence = None
                    # convert_one_hot_to_4d needs to be done to try to reduce dimensionality of input embedding
                    features_sequence = convert_sequence_to_4D(torch.tensor(
                        (data["sequence"]), dtype=torch.float32))
                    # print(features_sequence)
                    features_general = torch.tensor([v if k not in ('repetitive') else (1-v)*10 for k, v in data.items(
                    ) if k != 'gene' and k != 'exon' and k != 'position' and k != 'sequence'], dtype=torch.float32)
                    position = int(data.get("position", None))
                    if features_sequence is not None and features_general is not None:
                        features_general_list.append(features_general)
                        features_sequence_list.append(features_sequence)
                        # Add a mask of 1 for the sequence
                        mask_list.append(torch.ones(1, dtype=torch.bool))
                        positions_list.append(torch.tensor([position]))
                features_general_list = torch.stack(features_general_list)
                features_sequence_list = torch.stack(features_sequence_list)
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
                    # Pad the mask tensor
                    padded_mask = F.pad(
                        mask_list, (0, 0, 0, pad_size), 'constant', False)
                    padded_positions = F.pad(
                        positions_list, (0, 0, 0, pad_size), 'constant', -1)
                else:
                    padded_features_sequence = features_sequence_list
                    padded_features_general = features_general_list
                    padded_mask = mask_list
                    padded_positions = positions_list
                # Return the mask tensor
                yield padded_features_sequence, padded_features_general, padded_mask, padded_positions, current_file
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
        self.fc1 = nn.Linear(2*hidden_dim + normal_input_dim, hidden_dim)

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

        # Concatenate LSTM output and normal features
        combined_data = torch.cat((lstm_out, normal_data), dim=1)

        # Pass combined data through fully connected layers
        x = self.fc1(combined_data)
        x = self.dropout_fc1(x)
        x = self.fc2(x)

        # Apply sigmoid activation to get the probability
        x = torch.sigmoid(x)

        return x


def load_model(model_path):
    model = LSTMClassifier(seq_input_dim=4, normal_input_dim=3).to(device)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def identify_optimal_regions(predictions, threshold=0.1, seed_size=5):
    tags = [1 if p > threshold else 0 for p in predictions]
    
    seeds = []
    for i in range(0, len(tags) - seed_size + 1):
        if (tags[i:i+seed_size].count(1) / seed_size) >= threshold:
            seeds.append(i)

    regions = []
    for seed in seeds:
        start = seed
        end = seed + seed_size
        target_count = tags[start:end].count(1)
        total_count = end - start

        while start > 0 and (target_count + (tags[start-1] == 1)) / (total_count + 1) >= threshold:
            start -= 1
            total_count += 1
            target_count += tags[start] == 1

        while end < len(tags) and (target_count + (tags[end] == 1)) / (total_count + 1) >= threshold:
            end += 1
            total_count += 1
            target_count += tags[end-1] == 1

        regions.append((start, end))
    return regions

def calculate_regions(input_fasta_file, input_gff_file, model, threshold, seed_size, fragment_size, padding):
    result_df = pd.DataFrame(columns=['Filename', 'Record_ID', 'Strand', 'Start', 'End', 'Sequence'])
    with open(input_fasta_file, 'r') as fasta_file:
        for index, record in enumerate(SeqIO.parse(fasta_file, 'fasta')):
            len_sequence = len(record.seq)
            sequence = record.seq
            reverse_sequence = sequence.reverse_complement()
            temporary_fasta = f"{record.id}.fasta"
            SeqIO.write(record, temporary_fasta, "fasta")
            print(temporary_fasta)
            fordward_file, reverse_file = process_fasta(temporary_fasta, input_gff_file, output_file=temporary_fasta, min_length=50000, max_length=1000000, fragment_size=250, classification_mode=True)
            file_paths = [fordward_file, reverse_file]

            # Create datasets:
            forward_train_dataset = GeneDataset([fordward_file], mode="exon", shuffle=False)
            reverse_train_dataset = GeneDataset(
                [reverse_file], mode="exon", shuffle=False)
            # Dataloader
            forward_train_dataloader = DataLoader(
                forward_train_dataset, batch_size=10)
            reverse_train_dataloader = DataLoader(
                reverse_train_dataset, batch_size=10)

            forward_predictions = get_model_predictions_and_labels(model, forward_train_dataloader)
            reverse_predictions = get_model_predictions_and_labels(
                model, reverse_train_dataloader)

            forward_regions = identify_optimal_regions(
                forward_predictions, threshold, seed_size)
            reverse_regions = identify_optimal_regions(
                reverse_predictions, threshold, seed_size)

            forward_regions = [[start*fragment_size, end*fragment_size] for start, end in forward_regions]
            reverse_regions = [[start*fragment_size, end*fragment_size] for start, end in reverse_regions]

            padded_forward_regions = [
                (max(start-padding, 0), min(end+padding, len_sequence-1)) for start, end in forward_regions]
            padded_reverse_regions = [(max(start-padding,0), min(end+padding, len_sequence-1)) for start, end in reverse_regions]

            forward_sequences = [sequence[start:end] for start, end in padded_forward_regions]
            reverse_sequences = [reverse_sequence[start:end] for start, end in padded_reverse_regions]

            # Similar DataFrame creation and storage logic remains
            df_forward = pd.DataFrame({
                'Filename': [input_fasta_file for _ in forward_sequences],
                'Record_ID': [record.id for _ in forward_sequences],
                'Strand': ['forward' for _ in forward_sequences],
                'Start': [start+1 for start, end in padded_forward_regions],
                'End': [end+1 for start, end in padded_forward_regions],
                'Sequence': [str(seq) for seq in forward_sequences],
            })

            df_reverse = pd.DataFrame({
                'Filename': [input_fasta_file for _ in reverse_sequences],
                'Record_ID': [record.id for _ in reverse_sequences],
                'Strand': ['reverse' for _ in reverse_sequences],
                'Start': [len_sequence - start + 1 for start, end in padded_reverse_regions],
                'End': [len_sequence - end + 1 for start, end in padded_reverse_regions],
                'Sequence': [str(seq) for seq in reverse_sequences],
            })

            result_df = pd.concat([result_df, df_forward, df_reverse], ignore_index=True)
    return result_df


def get_model_predictions_and_labels(model, dataloader, threshold=0.5):
    model.eval()
    y_true = []
    y_pred = []
    rows = []
    torch.cuda.empty_cache()
    with torch.no_grad():
        for seq_data, normal_data,mask, positions, sequence_names in dataloader:
            print("Still living")
            seq_data, normal_data = seq_data.to(
                device), normal_data.to(device)
            seq_data = seq_data.view(-1, seq_data.size(2), seq_data.size(3))
            normal_data = normal_data.view(-1, normal_data.size(2))

            mask = mask.view(-1, 1)

            mask = mask.squeeze(-1).to(device).flatten()
            print(f"Size of seq_data: {tensor_size_in_MB(seq_data):.2f} MB")
            print(
                f"Size of normal_data: {tensor_size_in_MB(normal_data):.2f} MB")
            with profiler.profile(use_cuda=True) as prof:
                with profiler.record_function("model_inference"):
                    outputs = model(seq_data, normal_data).flatten()
            print(prof.key_averages().table(sort_by="cuda_time_total"))
            predicted_labels = (outputs > threshold).float()
            y_pred.extend(predicted_labels[mask].tolist())
            valid_positions = positions[mask].tolist()
            valid_sequence_names = sequence_names[mask].tolist()
            valid_predicted_scores = outputs[mask].tolist()
            torch.cuda.empty_cache()

    return y_pred

def get_genbank_record(result_df, input_fasta_file):
    new_records = []
    for record in SeqIO.parse(input_fasta_file, 'fasta'):
        new_record = SeqRecord(record.seq, id=record.id)
        for index, row in result_df.iterrows():
            if row['Record_ID'] == record.id:
                # Define the location of the feature
                location = FeatureLocation(
                    start=row['Start']-1, end=row['End'])

                # Create a new feature with the location and strand
                feature = SeqFeature(
                    location=location, strand=row['Strand'], type="Probable genetic region")

                # Create a SeqRecord object and append the feature to it

                new_record.features.append(feature)
        new_records.append(new_record)
    return new_records

def tensor_size_in_MB(tensor):
    return tensor.element_size() * tensor.nelement() / (1024 * 1024)

def main(args):
    model = load_model(args.model_path)
    result_df = calculate_regions(
        args.input_fasta_file, args.input_gff_file, model, args.threshold, args.seed_size, args.fragment_size, args.padding)
    output_filename = os.path.splitext(args.input_file)[0] + "_output.csv"
    # Save the DataFrame as a CSV file
    result_df.to_csv(os.path.join(
        args.output_directory, f'{args.input_fasta_file}_regions.csv'), index=False)
    genbank_records = get_genbank_record(
        result_df, args.input_fasta_file)
    SeqIO.write(genbank_records, os.path.join(
        args.output_directory, f'{args.input_fasta_file}_regions.gbk'), "genbank")
    print(f"Results saved to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identify regions using an LSTM model.")
    
    parser.add_argument("input_fasta_file", type=str, help="Path to the input FASTA file.")
    parser.add_argument('input_gff_file', type=str,
                        help='Input fasta file')
    parser.add_argument("model_path", type=str,
                        default="best_model_lstm.pth", help="Path to the LSTM model.")
    parser.add_argument('output_directory', type=str,
                        help='Output directory to store performance files')
    
    parser.add_argument("--threshold", type=float, default=0.75, help="Threshold for identifying optimal regions.")
    parser.add_argument("--seed_size", type=int, default=100, help="Seed size for identifying regions.")
    parser.add_argument("--fragment_size", type=int, default=100, help="Fragment size.")
    parser.add_argument("--padding", type=int, default=1500, help="padding value for regions.")
    
    args = parser.parse_args()
    
    main(args)
