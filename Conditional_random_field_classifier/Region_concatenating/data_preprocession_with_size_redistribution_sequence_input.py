import argparse
import json
import os

import gffutils
from Bio import SeqIO

ONE_HOT_ENCODING = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'T': [0, 0, 0, 1],
    'N': [0.25, 0.25, 0.25, 0.25],
    'R': [0.5, 0, 0.5, 0],
    'Y': [0, 0.5, 0, 0.5],
    'K': [0, 0, 0.5, 0.5],
    'M': [0.5, 0.5, 0, 0],
    'S': [0, 0.5, 0.5, 0],
    'W': [0.5, 0, 0, 0.5],
    'H':  [1/3, 1/3, 0, 1/3],
    'B':  [0, 1/3, 1/3, 1/3],
    'V': [1/3, 0, 1/3, 1/3],
    'D': [1/3, 1/3, 1/3, 0]
}

def is_repetitive(db, start, end, sequence_len, strand, record):
    if strand == -1:
        start, end = sequence_len - end, sequence_len - start

    # account for 1 based indexing in gff
    start += 1
    end += 1
    for feature in db.region(region=(record, start, end)):
        feature_id = feature.attributes.get('ID', None)
        if feature_id and 'repeat' in feature_id[0]:
            return 1

    return 0

def is_gene(db, start, end, sequence_len, strand, record):
    if strand == -1:
        start, end = sequence_len - end, sequence_len - start
    alternative_strand = "+" if strand == 1 else "-"
    # account for 1 based indexing in gff
    start += 1
    end += 1
    for feature in db.region(region=(record, start, end)):
        if (feature.featuretype == 'gene' or feature.featuretype == 'gene_quality') and (str(feature.strand) == str(strand) or feature.strand == alternative_strand):
            return 1
    return 0

def is_exon(db, start, end, sequence_len, strand, record):
    if strand == -1:
        start, end = sequence_len - end, sequence_len - start
    alternative_strand = "+" if strand == 1 else "-"
    # account for 1 based indexing in gff
    start += 1
    end += 1
    for feature in db.region(region=(record, start, end)):
        if (feature.featuretype == 'exon') and (feature.strand == strand or feature.strand == alternative_strand):
            return 1
    return 0

def one_hot_encode(sequence):

    encoded_sequence = [ONE_HOT_ENCODING[base] for base in sequence]
    return encoded_sequence


def process_fasta(fasta_file, gff_file, output_file=None, min_length=50000, max_length=1000000, fragment_size=250, classification_mode=False):
    # Open the GFF database
    db_path = gff_file + 'temp.db'


    # Check if the database file exists
    if not os.path.exists(db_path):
        # If it doesn't exist, create it
        db = gffutils.create_db(gff_file, dbfn=db_path, force=True,
                                keep_order=True, merge_strategy='merge',
                                sort_attribute_values=True)
    else:
        # If it exists, load it
        db = gffutils.FeatureDB(db_path)

    if output_file is not None:
        # Define output files for forward and reverse strands
        output_file_forward = output_file.rsplit('.', 1)[0] + '_forward.' + output_file.rsplit('.', 1)[1]
        output_file_reverse = output_file.rsplit('.', 1)[0] + '_reverse.' + output_file.rsplit('.', 1)[1]
        if output_file_forward is not None and os.path.exists(output_file_forward):
            os.remove(output_file_forward)
        if output_file_reverse is not None and os.path.exists(output_file_reverse):
            os.remove(output_file_reverse)

        with open(fasta_file, "r") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                # Separate forward and reverse sequence
                sequence_forward = str(record.seq)
                sequence_reverse = str(record.seq.reverse_complement())
                sequence_len = len(sequence_forward)

                # Skip sequences shorter than min_length
                #if sequence_len < min_length:
                #    continue

                # Process forward and reverse strand separately
                for strand, sequence, output_file_strand in zip([1, -1], [sequence_forward, sequence_reverse], [output_file_forward, output_file_reverse]):

                    with open(output_file_strand, 'a') as f:
                        # Adjusted range
                        for i in range(0, len(sequence) - fragment_size + 1, fragment_size):
                            start = i
                            end = i + fragment_size
                            fragment = sequence[start:end]
                            features = {}
                            features["sequence"] = one_hot_encode(fragment)
                            # Position in the original sequence
                            features['position'] = start
                            features['relative_position'] = start / \
                                sequence_len
                            # Forward strand = 1, Reverse strand = -1
                            features['strand'] = strand
                            features['repetitive'] = is_repetitive(
                                db, start, end, sequence_len, strand, record.id)
                            if classification_mode == False:
                                features['gene'] = is_gene(
                                    db, start, end, sequence_len, strand, record.id)
                                features['exon'] = is_exon(
                                    db, start, end, sequence_len, strand, record.id)
                            f.write(json.dumps(features) + '\n')
                print(record.id)


    return output_file_forward, output_file_reverse

def main():
    parser = argparse.ArgumentParser(description='Process fasta and gff files.')
    parser.add_argument('fasta_file', help='The fasta file to process.')
    parser.add_argument('gff_file', help='The GFF file to use for repeat regions.')
    parser.add_argument('output_file', help='The output file to write the fragments to.')
    parser.add_argument('--min_length', type=int, default=50000, help='Minimum length of sequences to process.')
    parser.add_argument('--max_length', type=int, default=1000000, help='Maximum length of sequences to process, otherwise the seuqences will be split.')
    parser.add_argument('--fragment_size', type=int, default=250, help='Fragment size to use when processing sequences.')
    args = parser.parse_args()

    process_fasta(args.fasta_file, args.gff_file, args.output_file, args.min_length, args.max_length, args.fragment_size)

if __name__ == '__main__':
    main()