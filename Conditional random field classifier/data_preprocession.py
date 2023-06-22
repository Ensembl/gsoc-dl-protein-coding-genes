import argparse
import gffutils
from Bio import SeqIO
from Bio.Seq import Seq
from collections import defaultdict
import os
import itertools

def count_kmers(sequence, k):
    d = defaultdict(int)

    # Initialize dictionary with all possible k-mers
    bases = ['A', 'C', 'G', 'T']
    all_kmers = [''.join(p) for p in itertools.product(bases, repeat=k)]
    for kmer in all_kmers:
        d[kmer] = 0

    # Count k-mers in sequence
    for i in range(len(sequence) - (k-1)):
        d[sequence[i:i+k]] += 1
    return dict(d)

def is_repetitive(db, start, end, sequence_len, strand):
    if strand == '-':
        start, end = sequence_len - end, sequence_len - start
    
    # account for 1 based indexing in gff
    start += 1
    end += 1
    for feature in db.region(region=(start, end)):
        if feature.featuretype == 'repeat':
            return 1
    return 0

def is_gene(db, start, end, sequence_len, strand):
    if strand == '-':
        start, end = sequence_len - end, sequence_len - start
    # account for 1 based indexing in gff
    start += 1
    end += 1
    for feature in db.region(region=(start, end)):
        if (feature.featuretype == 'gene' or feature.featuretype == 'gene_quality') and feature.strand == strand:
            return 1
    return 0

def process_fasta(fasta_file, gff_file, output_file = None, fragment_size=100, k=3, classification_mode = False):
    # Open the GFF database
    db = gffutils.create_db(gff_file, dbfn='temp.db', force=True,
                            keep_order=True, merge_strategy='merge',
                            sort_attribute_values=True)

    fragments = []
    with open(fasta_file, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            sequence = str(record.seq)
            sequence_rev = str(record.seq.reverse_complement())
            sequence_len = len(sequence)

            # Processing both normal and reverse complement strands
            fragments_in_record = []
            for seq, strand in [(sequence, '+'), (sequence_rev, '-')]:
                for i in range(0, sequence_len, fragment_size):
                    start = i
                    end = i + fragment_size
                    fragment = seq[start:end]
                    features = count_kmers(fragment, k)
                    features['position'] = start
                    features['relative_position'] = start/sequence_len
                    features['strand'] = 1 if strand == "+" else -1
                    features['repetitive'] = is_repetitive(db, start, end, sequence_len, strand)
                    if classification_mode == False:
                        features['gene'] = is_gene(db, start, end, sequence_len, strand)
                    fragments_in_record.append(features)
                fragments.append(fragments_in_record)

    # Remove temporary database
    os.remove('temp.db')

    if output_file is not None:
        # Write fragments to output file
        with open(output_file, 'w') as f:
            for fragment in fragments:
                f.write(str(fragment) + '\n')
    
    return fragments

def main():
    parser = argparse.ArgumentParser(description='Process fasta and gff files.')
    parser.add_argument('fasta_file', help='The fasta file to process.')
    parser.add_argument('gff_file', help='The GFF file to use for repeat regions.')
    parser.add_argument('output_file', help='The output file to write the fragments to.')
    args = parser.parse_args()

    process_fasta(args.fasta_file, args.gff_file, args.output_file)

if __name__ == '__main__':
    main()
