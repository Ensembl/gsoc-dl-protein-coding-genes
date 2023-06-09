import argparse
import gffutils
from Bio import SeqIO
from pyfaidx import Fasta
import os


def find_upstream_gene(sorted_genes, idx, seq_id):
    # Loop backwards from the current index
    for i in range(idx-1, -1, -1):
        # Return the first gene that shares the same seq_id
        if sorted_genes[i].seqid == seq_id:
            return sorted_genes[i]
    return None


def find_downstream_gene(sorted_genes, idx, seq_id):
    # Loop forwards from the current index
    for i in range(idx+1, len(sorted_genes)):
        # Return the first gene that shares the same seq_id
        if sorted_genes[i].seqid == seq_id:
            return sorted_genes[i]
    return None

def get_filename_from_path(path):
    base_name = os.path.basename(path)  # This will return file.ext
    file_name, ext = os.path.splitext(base_name)  # This will return ('file', '.ext')
    return file_name


def find_overlapping_genes(sorted_genes):
    print(sorted_genes)
    overlapping_genes = []
    for i in range(len(sorted_genes) - 1):
        # Check if the current gene overlaps with the next gene
        print(sorted_genes[i].seqid, sorted_genes[i + 1].seqid)
        if sorted_genes[i].end > sorted_genes[i + 1].start and sorted_genes[i].seqid == sorted_genes[i + 1].seqid:
            overlapping_genes.append(
                (sorted_genes[i].id, sorted_genes[i].seqid, (sorted_genes[i].start, sorted_genes[i].end)))
            overlapping_genes.append(
                (sorted_genes[i + 1].id, sorted_genes[i + 1].seqid, (sorted_genes[i + 1].start, sorted_genes[i + 1].end)))
    return overlapping_genes

def filter_sequences(output_dir, gff_file, fasta_file):

    # Derive the output filenames from the input filenames
    out_gff_file = os.path.join(output_dir, get_filename_from_path(gff_file) + '_filtered_removed_sequences.gff')
    out_fasta_file = os.path.join(output_dir, get_filename_from_path(fasta_file) + '_filtered_removed_sequences.fasta')

    # Create a database from the GFF file
    db = gffutils.create_db(gff_file, dbfn=os.path.splitext(gff_file)[0]+'temp.db', force=True, keep_order=True,
                            merge_strategy='merge', sort_attribute_values=True)

    # Get the ids and ranges of the low-quality genes
    low_quality_ids_ranges = [(feature.id, feature.seqid, (feature.start, feature.end))
                               for feature in db.all_features(order_by='start')
                              if (feature.featuretype == 'gene_quality' or feature.featuretype == 'gene') and 'gene_quality' in feature.attributes
                              and (feature.attributes['gene_quality'][0] == 'low' or feature.attributes['gene_quality'][0] == 'medium')]

    all_genes_sorted = sorted([feature for feature in db.all_features(
        order_by='start') if (feature.featuretype == 'gene_quality' or feature.featuretype == 'gene')], key=lambda x: (x.seqid, x.start))

    overlapping_gene_ids_ranges = find_overlapping_genes(all_genes_sorted)
    genes_to_remove = low_quality_ids_ranges.extend(overlapping_gene_ids_ranges)
    # Create a sorted list of all genes
    all_genes = sorted([feature for feature in db.all_features(
        order_by='start') if (feature.featuretype == 'gene_quality' or feature.featuretype == 'gene')], key=lambda x: x.start)

    # For each low-quality gene, find the upstream and downstream genes
    genes_to_remove_ranges_extended = []
    for id, seq_id, (start, end) in genes_to_remove:
        gene_idx = next(
            (i for i, g in enumerate(all_genes) if g.id == id), None)
        if gene_idx is not None:
            upstream_gene = find_upstream_gene(all_genes, gene_idx, seq_id)
            downstream_gene = find_downstream_gene(all_genes, gene_idx, seq_id)
            if upstream_gene:
                start_new = int((start + upstream_gene.end) / 2)
            else:
                start_new = int((start + 0) / 2)
            if downstream_gene:
                end_new = int((end + downstream_gene.start) / 2)
            else:
                end_new = end
            genes_to_remove_ranges_extended.append((id, seq_id, (start_new, end_new)))

    # Now, create a sorted list of all features, not just genes
    all_features = sorted([feature for feature in db.all_features(order_by='start')], key=lambda x: x.start)

    # Filter the GFF file
    with open(out_gff_file, 'w') as outfile:
        for feature in all_features:
            new_start = feature.start
            new_end = feature.end
            # Check if the feature is within the extended window of any low-quality gene
            for id, seq_id, (start, end) in genes_to_remove_ranges_extended:
                if seq_id == feature.seqid and not (feature.end < start or feature.start > end):
                    break  # This feature is within the extended window of a low-quality gene
            else:  # The 'else' clause of a 'for' loop is executed when the loop has exhausted the iterable
                # adjust the start and end points for the removed sequences
                for id, seq_id, (start, end) in genes_to_remove_ranges_extended:
                    if feature.start > end and seq_id == feature.seqid:
                        len_feature = end - start + 1
                        new_start -= len_feature
                        new_end -= len_feature
                feature.start = new_start
                feature.end = new_end
                outfile.write(str(feature) + '\n')

    # Filter the FASTA file
    with open(out_fasta_file, 'w') as outfile:
        for record in SeqIO.parse(fasta_file, "fasta"):
            print (record.id)
            last_end = 0
            new_sequence = ""
            for id, seq_id, (start, end) in sorted(genes_to_remove_ranges_extended, key=lambda x: x[1][0]):
                if seq_id == record.id:
                    # append the sequence before the current gene and after the last gene
                    new_sequence += str(record.seq[last_end:start-1])
                    last_end = end

            # append the sequence after the last gene
            new_sequence += str(record.seq[last_end:])
            outfile.write(f">{record.id}\n{new_sequence}\n")

    # Remove temporary database
    os.remove(os.path.splitext(gff_file)[0]+'temp.db')

def main():
    parser = argparse.ArgumentParser(description='Filter sequences and genes based on quality.')
    parser.add_argument('gff_file', help='The GFF file to filter.')
    parser.add_argument('fasta_file', help='The FASTA file to filter.')
    parser.add_argument('output_dir', help = "The output directory")
    args = parser.parse_args()
    filter_sequences(args.output_dir, args.gff_file, args.fasta_file)

if __name__ == '__main__':
    main()
