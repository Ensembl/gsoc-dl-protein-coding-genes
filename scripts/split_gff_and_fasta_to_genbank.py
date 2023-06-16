import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO
from Bio.SeqFeature import SeqFeature, FeatureLocation
from Bio import SeqRecord
import gffutils

def extract_regions(db, record, padding):
    features = list(db.features_of_type('gene_quality', order_by=('seqid', 'start', 'end')))
    regions = []
    for i, feature in enumerate(features):
        #print(feature.seqid, record.id)
        if feature.seqid != record.id:
            continue
        start = feature.start - 1
        end = feature.end
        #print (feature)
        if i > 0 and features[i - 1].seqid == record.id:
            start = max(start, features[i - 1].end + min((start - features[i - 1].end) // 2, padding))
        if i < len(features) - 1 and features[i + 1].seqid == record.id:
            end = min(end, features[i + 1].start - min((features[i + 1].start - end) // 2, padding))
        regions.append((start, end))
    return regions

def extract_intergenic_regions(record, regions):
    starts = sorted(end for _, end in regions)
    ends = sorted(start for start, _ in regions)
    if len(starts) == 0:
        starts.insert(0, 0)
        ends.append(len(record))
        return [(start, end) for start, end in zip(starts, ends) if start != end]
    if starts[0] > 0:
        starts.insert(0, 0)
    if ends[-1] < len(record):
        ends.append(len(record))
    return [(start, end) for start, end in zip(starts, ends) if start != end]


def create_genbank_files(fasta_file, gff_file, output_dir):
    db = gffutils.create_db(gff_file, ':memory:')
    df_list = []  # This list will hold the data for our DataFrame
    counter = 0
    with open(fasta_file, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            counter += 1
            regions = extract_regions(db, record, 1500)
            lengths = [end - start for start, end in regions]
            if not lengths:  # if lengths list is empty
                # generating normally distributed lengths
                lengths = [abs(int(value)) for value in np.random.normal(20000, 10000, 10).tolist()]
            for i, (start, end) in enumerate(regions):
                filename = f"{record.id}_region{i+1}.gbk"
                new_record = SeqRecord.SeqRecord(
                    record.seq[start:end], id=f"{record.id}_region{i+1}", description="")
                # add the molecule type
                new_record.annotations["molecule_type"] = "DNA"
                new_record.features.append(SeqFeature(
                    FeatureLocation(min(int(start), int(end)), max(int(start), int(end)), strand=1), type="source"))
                with open(os.path.join(output_dir, filename), "w") as output_handle:
                    SeqIO.write(new_record, output_handle, "genbank")
                df_list.append({'sequence': str(
                    record.seq[start:end]), 'filename': filename, 'target': 1, 'start': start, 'end': end})

            intergenic_regions = extract_intergenic_regions(record, regions)
            for i, (start, end) in enumerate(intergenic_regions):
                j = start
                while j < end:
                    chunk_length = np.random.choice(
                        lengths)  # randomly select a length
                    if chunk_length > end - j:  # if the randomly selected length exceeds the length of the remaining intergenic region, adjust it
                        chunk_length = end - j
                    filename = f"{record.id}_intergenic{i+1}.gbk"
                    new_record = SeqRecord.SeqRecord(
                        record.seq[j:j + chunk_length], id=f"{record.id}_intergenic{i+1}", description="")
                    # add the molecule type
                    new_record.annotations["molecule_type"] = "DNA"
                    new_record.features.append(SeqFeature(FeatureLocation(min(int(j), int(j + chunk_length)), max(int(j), int(j + chunk_length)), strand=1), type="source"))
                    with open(os.path.join(output_dir, filename), "w") as output_handle:
                        SeqIO.write(new_record, output_handle, "genbank")
                    df_list.append({'sequence': str(
                        record.seq[j:j + chunk_length]), 'filename': filename, 'target': 0, 'start': np.nan, 'end': np.nan})
                    j += chunk_length  # move to the next chunk
    # Create the DataFrame
    df = pd.DataFrame(df_list)
    # Save the DataFrame to a CSV file
    df.to_csv(os.path.join(output_dir, 'fragemented_files.csv'), index=False)

    return df

def plot_data(df, output_dir):
    # Calculate sequence lengths
    df['length'] = df['sequence'].apply(len)

    # Plot histograms for the lengths of genetic and intergenic regions
    fig, ax = plt.subplots(3, 1, sharex=False, figsize=(12, 8))

    ax[0].hist(df.loc[df['target'] == 1, 'length'],
               bins=30, color='b', alpha=0.7)
    ax[0].set_title('Length distribution of genetic regions')

    ax[1].hist(df.loc[df['target'] == 0, 'length'],
               bins=30, color='r', alpha=0.7)
    ax[1].set_title('Length distribution of intergenic regions')

    # Check if 'start' column contains any non-null values before plotting
    if df['start'].notna().any():
        ax[2].hist(df.loc[df['target'] == 1, 'start'].dropna(),
                   bins=30, color='g', alpha=0.7)
        ax[2].set_title('Distribution of gene start positions')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'histograms.png'))


def main():
    parser = argparse.ArgumentParser(description='Create GenBank files for each gene and its surrounding region.')
    parser.add_argument('gff_file', help='Input GFF file.')
    parser.add_argument('fasta_file', help='Input FASTA file.')
    parser.add_argument('output_dir', help='Output directory')
    args = parser.parse_args()
    df = create_genbank_files(args.fasta_file, args.gff_file, args.output_dir)
    plot_data(df, args.output_dir)

if __name__ == '__main__':
    main()
