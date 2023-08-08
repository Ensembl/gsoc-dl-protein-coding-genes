import argparse
import random
import gffutils
import Bio.SeqIO as SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import os


def generate_random_sequence(length):
    return ''.join(random.choice('ACGT') for _ in range(length))


# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Generate a fasta and GFF with alternating exon and intergenic/random sequences.")
parser.add_argument('--fasta', required=True, help='Input fasta file')
parser.add_argument('--gff', required=True, help='Input GFF file')
parser.add_argument('--random', action='store_true',
                    help='Use random sequences instead of intergenic regions')
args = parser.parse_args()

# Database creation
db = gffutils.create_db(args.gff, dbfn='temp.db', force=True,
                        keep_order=True, merge_strategy='merge', sort_attribute_values=True)

# Open your fasta file
fasta_sequences = SeqIO.to_dict(SeqIO.parse(open(args.fasta), 'fasta'))

new_fasta_records = []
new_gff_records = []

# Get all exon features sorted by start position
exons = sorted([exon for exon in db.features_of_type('exon')
               if exon.strand == "1" or exon.strand == "+"], key=lambda x: x.start)
print(set([exon.seqid for exon in exons]))
print(fasta_sequences.keys())
fasta_records={k:"" for k in fasta_sequences.keys()}
print(fasta_records)
for i in range(len(exons) - 1):
    try:
        exon = exons[i]
        next_exon = exons[i + 1]

        exon_seq = fasta_sequences[exon.seqid].seq[exon.start-1:exon.end]

        if len(exon_seq) >= 250:
            fasta_records[exon.seqid] += str(exon_seq)

        if args.random:
            intergenic_seq = generate_random_sequence(250)
        else:
            intergenic_seq = fasta_sequences[exon.seqid].seq[exon.end:next_exon.start-1]
            
        if len(intergenic_seq) >= 250:
                fasta_records[exon.seqid] += str(intergenic_seq)
                new_gff_records.append(
                    f"{exon.seqid}\t.\tintergenic\t{exon.end+1}\t{exon.end+250}\t.\t{exon.strand}\t.\tID=intergenic_{exon.id};")
                # Find all features within the intergenic region
                features_in_intergenic = db.region(
                    region=(exon.seqid, exon.end, exon.end+250), completely_within=False)
                for feature in features_in_intergenic:
                    new_gff_records.append(str(feature))

    except:
        print ("error")
print(fasta_records)
for key, value in fasta_records:
    new_fasta_records.append(SeqRecord(Seq(value), id =key))
# Generate output file names
fasta_out = os.path.splitext(args.fasta)[0] + "_output.fasta"
gff_out = os.path.splitext(args.gff)[0] + "_output.gff"

# Write new fasta file
SeqIO.write(new_fasta_records, fasta_out, 'fasta')

# Write new GFF file
with open(gff_out, 'w') as output_gff:
    output_gff.write("\n".join(new_gff_records))
