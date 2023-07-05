import argparse
import gffutils
from Bio import SeqIO
from Bio.Seq import Seq
from collections import defaultdict
import os
import itertools
import multiprocessing
from functools import partial

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

def is_repetitive(db, start, end, sequence_len, strand, seq_id):
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

def is_gene(db, start, end, sequence_len, strand, seq_id):
    if strand == -1:
        start, end = sequence_len - end, sequence_len - start
    # account for 1 based indexing in gff
    start += 1
    end += 1
    for feature in db.region(region=(seq_id, start, end)):
        #print(strand, feature.strand, feature.featuretype)
        if (feature.featuretype == 'gene' or feature.featuretype == 'gene_quality') and int(feature.strand) == int(strand):
            print ("found_gene")
            return 1
    return 0


def process_record(record, index, db, fragment_size, k, classification_mode):
    sequence = str(record.seq)
    sequence_rev = str(record.seq.reverse_complement())
    sequence_len = len(sequence)

    # Processing both normal and reverse complement strands
    fragments_in_record = []
    for seq, strand in [(sequence, +1), (sequence_rev, -1)]:
        for i in range(0, sequence_len, fragment_size):
            start = i
            end = i + fragment_size
            fragment = seq[start:end]
            features = count_kmers(fragment, k)
            features['position'] = start
            features['relative_position'] = start/sequence_len
            features['strand'] = strand
            features['repetitive'] = is_repetitive(
                db, start, end, sequence_len, strand, record.id)
            if classification_mode == False:
                features['gene'] = is_gene(
                    db, start, end, sequence_len, strand, record.id)
            fragments_in_record.append(features)

    return index, record.id, fragments_in_record


def process_fasta(fasta_file, gff_file, output_file=None, fragment_size=1000, k=3, classification_mode=False):
    # Open the GFF database
    print(gff_file)
    db = gffutils.create_db(gff_file, dbfn=f'{gff_file}_temp.db', force=True,
                            keep_order=True, merge_strategy='merge',
                            sort_attribute_values=True)
    records = [(record, i)
               for i, record in enumerate(SeqIO.parse(fasta_file, "fasta"))]

    fragments = []
    if output_file is not None:
        with open(output_file, 'w') as f:
            with open(fasta_file, "r") as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    print (record)
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
                            features['repetitive'] = is_repetitive(db, start, end, sequence_len, strand, record.id)
                            if classification_mode == False:
                                features['gene'] = is_gene(db, start, end, sequence_len, strand, record.id)
                            fragments_in_record.append(features)
                    fragments.append(fragments_in_record)
                    f.write(str(fragments_in_record) + '\n')

    with open(output_file, 'w') as f:
        for index, record_id, fragments_in_record in fragments:
            f.write(str(fragments_in_record) + '\n')

    os.remove(f'{gff_file}_temp.db')

    return fragments

def main():
    parser = argparse.ArgumentParser(description='Process fasta and gff files.')
    parser.add_argument('fasta_file', help='The fasta file to process.')
    parser.add_argument('gff_file', help='The GFF file to use for repeat regions.')
    parser.add_argument('output_file', help='The output file to write the fragments to.')
    args = parser.parse_args()

    process_fasta(args.fasta_file, args.gff_file, output_file = args.output_file)

if __name__ == '__main__':
    main()
