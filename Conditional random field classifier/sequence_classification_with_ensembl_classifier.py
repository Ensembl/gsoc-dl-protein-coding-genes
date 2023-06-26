from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC
import pycrfsuite
import joblib
from data_preprocession import process_fasta
import pandas as pd
import glob
import ast
import multiprocessing
import pycrfsuite
import numpy as np
import os
import argparse
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, FeatureLocation
from Bio import SeqIO

# You can set the seed size to any value you want
SEED_SIZE = 100  # for example
THRESHOLD = 0.75
PADDING = 1500


def model_prediction(model_file, weights, data):
    # Load the model
    model = pycrfsuite.Tagger()
    model.open(model_file)
    # Compute the predictions
    predictions = model.tag(data)
    # Convert the predictions to probabilities (assuming binary classification with 'gene' as the positive class)
    probabilities = [1 if prediction ==
                     'gene' else 0 for prediction in predictions]
    # Multiply the probabilities with the weight of that model
    weighted_probabilities = np.array(probabilities) * weights[model_file]

    return weighted_probabilities


def weighted_prediction(data, model_files, output_directory):
    # Initialize a dictionary to store the weights
    weights = {}

    # Load the performance metrics for each model
    for model_file in model_files:
        # Get the base filename without extension
        filename_base = os.path.basename(model_file).split('.')[0]
        # Extract the hyperparameters from the filename
        _, _, c1, _, c2, _ = filename_base.split('_')
        # Generate the name of the performance file
        performance_file = os.path.join(
            output_directory, f'hyperparameters_{filename_base}.txt')
        # Open the performance file and load the metrics
        with open(performance_file, 'r') as f:
            performance_metrics_dict = ast.literal_eval(f.read())
        # Extract the performance metrics for the current hyperparameters
        performance_metrics = performance_metrics_dict[(float(c1), float(c2))]
        # Compute the weight for this model (e.g., based on the balanced accuracy)
        weights[model_file] = performance_metrics['balanced_accuracy']

    # Compute the sum of the weights (for normalization)
    total_weight = sum(weights.values())

    # Normalize the weights
    weights = {k: v / total_weight for k, v in weights.items()}

    # Using multiprocessing for parallel prediction
    with multiprocessing.Pool() as pool:
        weighted_probabilities_list = pool.starmap(
            model_prediction, [(model_file, weights, data) for model_file in model_files])

    # Sum all the weighted predictions to get the final prediction
    weighted_predictions = np.sum(weighted_probabilities_list, axis=0)

    # Convert the weighted predictions to labels (assuming a threshold of 0.5)
    weighted_labels = ['gene' if prediction >=
                       0.5 else 'no-gene' for prediction in weighted_predictions]

    return weighted_labels

def identify_optimal_regions(tags, target_tag, threshold, seed_size):
    # First, identify seeds
    seeds = []
    for i in range(0, len(tags) - seed_size + 1):
        if (tags[i:i+seed_size].count(target_tag) / seed_size) >= threshold:
            seeds.append(i)

    # Next, expand each seed into a region
    regions = []
    for seed in seeds:
        # Initialize the region to the seed
        start = seed
        end = seed + seed_size
        target_count = tags[start:end].count(target_tag)
        total_count = end - start

        # Expand the region to the left
        while start > 0 and (target_count + (tags[start-1] == target_tag)) / (total_count + 1) >= threshold:
            start -= 1
            total_count += 1
            target_count += tags[start] == target_tag

        # Expand the region to the right
        while end < len(tags) and (target_count + (tags[end] == target_tag)) / (total_count + 1) >= threshold:
            end += 1
            total_count += 1
            target_count += tags[end-1] == target_tag

        regions.append((start, end))

    return regions


def calculate_regions(input_fasta_file, input_gff_file, model_files, input_directory, fragment_size=100):
    # Initialize an empty DataFrame to store all the results
    result_df = pd.DataFrame(columns=['Filename', 'Record_ID', 'Strand', 'Start', 'End', 'Sequence'])

    # Open the FASTA file and read in the sequence
    features = process_fasta(input_fasta_file, input_gff_file, output_file=None,
                             fragment_size=1000, k=3, classification_mode=True)

    with open(input_fasta_file, 'r') as fasta_file:
        for index, record in enumerate(SeqIO.parse(fasta_file, 'fasta')):
            sequence = record.seq
            len_sequence = len(sequence)
            reverse_sequence = sequence.reverse_complement()
            features_record = features[index]
        
            forward_features, reverse_features = features_record[:len(features_record)//2], features_record[len(features_record)//2:]

            # Process the forward strand
            forward_tags = weighted_prediction(
                forward_features, model_files, input_directory)

            # Process the reverse strand
            reverse_tags = weighted_prediction(
                reverse_features, model_files, input_directory)

            # Identify regions where 'gene2' frequency is above threshold
            forward_regions_tags = identify_optimal_regions(forward_tags, 1, threshold=THRESHOLD)
            reverse_regions_tags = identify_optimal_regions(reverse_tags, 1, threshold=THRESHOLD)

            # Account for the fact that the tags have a specific size
            forward_regions = [[start*fragment_size, end*fragment_size] for start, end in forward_regions_tags]
            reverse_regions = [[start*fragment_size, end*fragment_size] for start, end in reverse_regions_tags]

            # Add padding to these regions
            padded_forward_regions = [(max(start-PADDING,0), min(end+PADDING, len_sequence-1)) for start, end in forward_regions]
            padded_reverse_regions = [(max(start-PADDING,0), min(end+PADDING, len_sequence-1)) for start, end in reverse_regions]

            # Extract sequences for these regions
            forward_sequences = [sequence[start:end] for start, end in padded_forward_regions]
            reverse_sequences = [reverse_sequence[start:end] for start, end in padded_reverse_regions]

            # Create a DataFrame for forward sequences
            # The +1 is to adjust for GFF 1 based indexing
            df_forward = pd.DataFrame({
                'Filename': [input_fasta_file for _ in forward_sequences],
                'Record_ID': [record.id for _ in forward_sequences],
                'Strand': [1 for _ in forward_sequences],
                'Start': [start+1 for start, end in padded_forward_regions],
                'End': [end+1 for start, end in padded_forward_regions],
                'Sequence': [str(seq) for seq in forward_sequences],
            })

            # Create a DataFrame for reverse sequences
            df_reverse = pd.DataFrame({
                'Filename': [input_fasta_file for _ in reverse_sequences],
                'Record_ID': [record.id for _ in reverse_sequences],
                'Strand': [-1 for _ in reverse_sequences],
                'Start': [len_sequence - start + 1 for start, _ in padded_reverse_regions],
                'End': [len_sequence - end + 1 for _, end in padded_reverse_regions],
                'Sequence': [str(seq) for seq in reverse_sequences],
            })

            # Concatenate both DataFrames
            result_df = pd.concat([result_df, df_forward, df_reverse], ignore_index=True)

    return result_df


def get_genbank_record(
    result_df, input_fasta_file):
    new_records = []
    for record in SeqIO.parse(input_fasta_file, 'fasta'):
        new_record = SeqRecord(record.seq, id=record.id)
        for index, row in result_df.iterrows():
            if row['Record_ID'] == record.id:
                # Define the location of the feature
                location = FeatureLocation(start=row['Start']-1, end=row['End'])

                # Create a new feature with the location and strand
                feature = SeqFeature(
                    location=location, strand=row['Strand'], type="Probable genetic region")
                
                # Create a SeqRecord object and append the feature to it

                new_record.features.append(feature)
        new_records.append(new_record)
    return new_records
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Perform ensemble prediction on a fasta file')
    parser.add_argument('input_directory', type=str,
                        help='Input directory with classifiers')
    parser.add_argument('input_fasta_file', type=str,
                        help='Input fasta file')
    parser.add_argument('input_gff_file', type=str,
                        help='Input fasta file')
    parser.add_argument('output_directory', type=str,
                        help='Output directory to store performance files')
    args = parser.parse_args()

    # Get a list of all the model files in the input directory
    input_fasta_file = args.input_fasta_file
    input_directory = args.input_directory
    output_directory = args.output_directory
    model_files = glob.glob(os.path.join(input_directory, '*.crfsuite'))
    result_df = calculate_regions(
        input_fasta_file, model_files, input_directory, fragment_size=100)
    # Save the DataFrame as a CSV file
    result_df.to_csv(os.path.join(
        args.output_directory, f'{input_fasta_file}_regions.csv'), index=False)
    genbank_records = get_genbank_record(
        result_df, input_fasta_file)
    SeqIO.write(genbank_records, os.path.join(
        args.output_directory, f'{input_fasta_file}_regions.gbk'), "genbank")

            
