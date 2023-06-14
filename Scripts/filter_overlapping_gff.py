import argparse
import gffutils
import os

def find_overlapping_genes(sorted_genes):
    print(sorted_genes)
    overlapping_genes = []
    for i in range(len(sorted_genes) - 1):
        # Check if the current gene overlaps with the next gene
        print(sorted_genes[i].seqid, sorted_genes[i +1].seqid)
        if sorted_genes[i].end > sorted_genes[i + 1].start and sorted_genes[i].seqid == sorted_genes[i + 1].seqid:
            overlapping_genes.append(
                (sorted_genes[i].id, sorted_genes[i].seqid, (sorted_genes[i].start, sorted_genes[i].end)))
            overlapping_genes.append(
                (sorted_genes[i + 1].id, sorted_genes[i + 1].seqid, (sorted_genes[i + 1].start, sorted_genes[i + 1].end)))
    return overlapping_genes

def get_filename_from_path(path):
    base_name = os.path.basename(path)  # This will return file.ext
    file_name, ext = os.path.splitext(base_name)  # This will return ('file', '.ext')
    return file_name

def filter_genes(output_dir, input_file):
    # Derive the output filename from the input filename
    output_file = os.path.join(output_dir, get_filename_from_path(
        input_file) + '_filtered_annotations_overlapping_genes.gff')

    # Create a database from the GFF file
    db = gffutils.create_db(input_file, dbfn=f'{get_filename_from_path(input_file)}temp.db', force=True, keep_order=True, merge_strategy='merge', sort_attribute_values=True)

    # Get the ids and ranges of the low-quality genes
    all_genes_sorted = sorted([feature for feature in db.all_features(
        order_by='start') if feature.featuretype == 'gene_quality'], key=lambda x: (x.seqid, x.start))

    overlapping_gene_ids_ranges = find_overlapping_genes(all_genes_sorted)
    all_genes = sorted([feature for feature in db.all_features(
        order_by='start') if feature.featuretype == 'gene_quality'], key=lambda x: x.start)

    with open(output_file, 'w') as outfile:
        for feature in all_genes:

            if feature.id not in [id for id, _, _ in overlapping_gene_ids_ranges]:
                outfile.write(str(feature) + '\n')

    # Remove temporary database
    os.remove(f'{get_filename_from_path(input_file)}temp.db')

def main():
    parser = argparse.ArgumentParser(description='Filter genes based on quality.')
    parser.add_argument('input_file', help='The GFF file to filter.')
    parser.add_argument('output_dir', help='The ouput directory for the files.')
    args = parser.parse_args()
    filter_genes(args.output_dir, args.input_file)

if __name__ == '__main__':
    main()
