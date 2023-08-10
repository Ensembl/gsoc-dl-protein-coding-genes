import argparse
import pandas as pd
import matplotlib.pyplot as plt
from Bio import SeqIO
from BCBio import GFF
from collections import Counter

def analyze_regions_and_genes(df_file, fasta_file, gff_file, output_file):
    # Read DataFrame
    df = pd.read_csv(df_file)
    
    # Read the genomic sequence
    genomic_records = SeqIO.to_dict(SeqIO.parse(fasta_file, 'fasta'))

    # Read the gene annotations
    limit_info = dict(gff_type = ['gene'])
    gff_iter = GFF.parse(gff_file, limit_info=limit_info, base_dict=genomic_records)

    # Initialize counters for genes and padding
    gene_counters = {'High': {'Forward': Counter(), 'Reverse': Counter()},
                     'Medium': {'Forward': Counter(), 'Reverse': Counter()},
                     'Low': {'Forward': Counter(), 'Reverse': Counter()}}
    
    region_counters = {'High': {'Forward': Counter(), 'Reverse': Counter()},
                       'Medium': {'Forward': Counter(), 'Reverse': Counter()},
                       'Low': {'Forward': Counter(), 'Reverse': Counter()}}
    
    padding = {'High': {'Forward': {'Upstream': [], 'Downstream': []}, 'Reverse': {'Upstream': [], 'Downstream': []}},
               'Medium': {'Forward': {'Upstream': [], 'Downstream': []}, 'Reverse': {'Upstream': [], 'Downstream': []}},
               'Low': {'Forward': {'Upstream': [], 'Downstream': []}, 'Reverse': {'Upstream': [], 'Downstream': []}}}

    # Loop over each gene in the GFF3 file
    for record in gff_iter:
        for feature in record.features:
            gene = feature
            gene_quality = gene.qualifiers['gene_quality'][0] 
            strand = 'Forward' if gene.location.strand == 1 else 'Reverse'

            # Loop over each region in the DataFrame
            for _, region in df[df['Strand'] == strand].iterrows():
                # Check if the gene is located within the region
                if region['Start'] <= gene.location.start and region['End'] >= gene.location.end:
                    gene_counters[gene_quality][strand]['completely contained'] += 1
                    region_counters[gene_quality][strand]['one gene'] += 1
                    padding[gene_quality][strand]['Upstream'].append(gene.location.start - region['Start'])
                    padding[gene_quality][strand]['Downstream'].append(region['End'] - gene.location.end)
                elif region['Start'] <= gene.location.start < region['End'] or region['Start'] < gene.location.end <= region['End']:
                    gene_counters[gene_quality][strand]['partially contained'] += 1
                    region_counters[gene_quality][strand]['multiple genes'] += 1
                else:
                    gene_counters[gene_quality][strand]['not contained'] += 1
                    region_counters[gene_quality][strand]['no gene'] += 1

    # Reset padding for regions with multiple genes
    for quality, strands in region_counters.items():
        for strand, counter in strands.items():
            if counter['multiple genes'] > 0:
                padding[quality][strand]['Upstream'] = []
                padding[quality][strand]['Downstream'] = []

    # Open the output file
    with open(output_file, 'w') as f:
        # Plot pie charts for each gene quality and strand
        for quality, strands in gene_counters.items():
            for strand, counter in strands.items():
                plt.figure()
                plt.pie(counter.values(), labels=counter.keys(), autopct='%1.1f%%')
                plt.title(f'Gene Locations for {quality} Quality Genes on {strand} Strand')
                plt.savefig(f, format='png')

        # Plot histograms for each gene quality and strand
        for quality, strands in padding.items():
            for strand, dirs in strands.items():
                for direction, data in dirs.items():
                    plt.figure()
                    plt.hist(data, bins=30, alpha=0.5)
                    plt.title(f'{direction} Padding for {quality} Quality Genes on {strand} Strand')
                    plt.xlabel('Padding')
                    plt.ylabel('Frequency')
                    plt.savefig(f, format='png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze regions and genes.')
    parser.add_argument('-d', '--dataframe', required=True, help='The DataFrame CSV file.')
    parser.add_argument('-f', '--fasta', required=True, help='The FASTA file.')
    parser.add_argument('-g', '--gff', required=True, help='The GFF3 file.')
    parser.add_argument('-o', '--output', required=True, help='The output file.')
    args = parser.parse_args()

    analyze_regions_and_genes(args.dataframe, args.fasta, args.gff, args.output)
