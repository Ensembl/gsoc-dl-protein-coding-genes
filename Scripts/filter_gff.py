import argparse
import gffutils
import os

def filter_genes(input_file):
    # Derive the output filename from the input filename
    output_file = os.path.splitext(input_file)[0] + '_filtered.gff'

    # Create a database from the GFF file
    db = gffutils.create_db(input_file, dbfn='temp.db', force=True, keep_order=True,
                            merge_strategy='merge', sort_attribute_values=True)

    # Create an output GFF file
    with open(output_file, 'w') as outfile:
        # Iterate over each feature in the database
        for feature in db.all_features(order_by='start'):
            #print(feature.attributes, feature.featuretype)
            # Check if the feature is a gene and if the quality is high
            if feature.featuretype == 'gene_quality' and 'gene_quality' in feature.attributes:
                #print(feature.attributes['gene_quality'][0])
                if feature.attributes['gene_quality'][0] == 'high':
                    # Write the gene to the output file
                    outfile.write(str(feature) + '\n')

    # Remove temporary database
    os.remove('temp.db')

def main():
    parser = argparse.ArgumentParser(description='Filter genes based on quality.')
    parser.add_argument('input_file', help='The GFF file to filter.')
    args = parser.parse_args()
    filter_genes(args.input_file)

if __name__ == '__main__':
    main()
