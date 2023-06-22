from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC
import pycrfsuite
import joblib
from data_preprocession import process_fasta
import pandas as pd

# You can set the seed size to any value you want
SEED_SIZE = 100  # for example
THRESHOLD = 0.75
PADDING = 1500
# Load the model
tagger = pycrfsuite.Tagger()
tagger.open('best_model.crfsuite')  # replace 'best_model.crfsuite' with your model file


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


def calculate_regions(input_file, model, fragment_size=100):
    # Initialize an empty DataFrame to store all the results
    result_df = pd.DataFrame(columns=['Filename', 'Record_ID', 'Strand', 'Start', 'End', 'Sequence'])

    # Open the FASTA file and read in the sequence
    features = process_fasta(input_file)

    with open(input_file, 'r') as fasta_file:
        for index, record in enumerate(SeqIO.parse(fasta_file, 'fasta')):
            sequence = record.seq
            len_sequence = len(sequence)
            reverse_sequence = sequence.reverse_complement()
            features_record = features[index]
        
            forward_features, reverse_features = features_record[:len(features_record)//2], features_record[len(features_record)//2:]

            # Process the forward strand
            forward_tags = tagger.tag(forward_features)

            # Process the reverse strand
            reverse_tags = tagger.tag(reverse_features)

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
                'Filename': [input_file for _ in forward_sequences],
                'Record_ID': [record.id for _ in forward_sequences],
                'Strand': ['forward' for _ in forward_sequences],
                'Start': [start+1 for start, end in padded_forward_regions],
                'End': [end+1 for start, end in padded_forward_regions],
                'Sequence': [str(seq) for seq in forward_sequences],
            })

            # Create a DataFrame for reverse sequences
            df_reverse = pd.DataFrame({
                'Filename': [input_file for _ in reverse_sequences],
                'Record_ID': [record.id for _ in reverse_sequences],
                'Strand': ['reverse' for _ in reverse_sequences],
                'Start': [len_sequence - start + 1 for start, end in padded_reverse_regions],
                'End': [len_sequence - end + 1 for start, end in padded_reverse_regions],
                'Sequence': [str(seq) for seq in reverse_sequences],
            })

            # Concatenate both DataFrames
            result_df = pd.concat([result_df, df_forward, df_reverse], ignore_index=True)

    # Save the DataFrame as a CSV file
    result_df.to_csv('output.csv', index=False)

    return result_df


            
