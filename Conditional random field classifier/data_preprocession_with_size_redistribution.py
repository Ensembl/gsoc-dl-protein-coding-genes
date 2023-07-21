import argparse
import json
import gffutils
from Bio import SeqIO
from Bio.Seq import Seq
from collections import defaultdict
import os
import itertools

def count_kmers(sequence, k):
    d = defaultdict(int)

    # Initialize dictionary with all possible k-mers
    bases = ['A', 'C', 'G', 'T', "N"]
    all_kmers = [''.join(p) for p in itertools.product(bases, repeat=k)]
    for kmer in all_kmers:
        d[kmer] = 0

    # Count k-mers in sequence
    for i in range(len(sequence) - (k-1)):
        if sequence[i:i+k] in all_kmers:
            d[sequence[i:i+k]] += 1
    return dict(d)

def is_repetitive(db, start, end, sequence_len, strand, record):
    if strand == -1:
        start, end = sequence_len - end, sequence_len - start

    # account for 1 based indexing in gff
    start += 1
    end += 1
    for feature in db.region(region=(seq_id, start, end)):
        feature_id = feature.attributes.get('ID', None)
        if feature_id and 'repeat' in feature_id:
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
        if (feature.featuretype == 'gene' or feature.featuretype == 'gene_quality' or feature.featuretype == 'exon') and (feature.strand == strand or feature.strand == alternative_strand):
            return 1
    return 0

def process_fasta(fasta_file, gff_file, output_file=None, min_length=50000, max_length=1000000, fragment_size=250, k=3, classification_mode=False):
    # Open the GFF database
    db = gffutils.create_db(gff_file, dbfn=fasta_file+'temp.db', force=True,
                            keep_order=True, merge_strategy='merge',
                            sort_attribute_values=True)

    fragments = []
    if output_file is not None:
        # Define output files for forward and reverse strands
        output_file_forward = output_file.rsplit('.', 1)[0] + '_forward.' + output_file.rsplit('.', 1)[1]
        output_file_reverse = output_file.rsplit('.', 1)[0] + '_reverse.' + output_file.rsplit('.', 1)[1]

        with open(fasta_file, "r") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                # Separate forward and reverse sequence
                sequence_forward = str(record.seq)
                sequence_reverse = str(record.seq.reverse_complement())
                sequence_len = len(sequence_forward)

                # Skip sequences shorter than min_length
                if sequence_len < min_length:
                    continue

                # Process forward and reverse strand separately
                for strand, sequence, output_file_strand in zip([1, -1], [sequence_forward, sequence_reverse], [output_file_forward, output_file_reverse]):
                    # Split sequences longer than max_length
                    sequences = [sequence[i: i+max_length] for i in range(0, len(sequence), max_length)]

                    with open(output_file_strand, 'a') as f:
                        for seq_idx, seq in enumerate(sequences):
                            fragments_in_record = []
                            for i in range(0, len(seq) - fragment_size + 1, fragment_size):  # Adjusted range
                                global_start = seq_idx * max_length  # Starting position of the sub-sequence in the original sequence
                                start = i
                                end = i + fragment_size
                                fragment = seq[start:end]
                                features = count_kmers(fragment, k)
                                features['position'] = global_start + start  # Position in the original sequence
                                features['relative_position'] = (global_start + start) / sequence_len
                                features['strand'] = strand  # Forward strand = 1, Reverse strand = -1
                                features['repetitive'] = is_repetitive(db, global_start + start, global_start + end, sequence_len, strand == strand, record.id)
                                if classification_mode == False:
                                    features['gene'] = is_gene(db, global_start + start, global_start + end, sequence_len, strand == strand, record.id)
                                fragments_in_record.append(features)
                            f.write(json.dumps(fragments_in_record) + '\n')

    # Remove temporary database
    os.remove(fasta_file+'temp.db')

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
