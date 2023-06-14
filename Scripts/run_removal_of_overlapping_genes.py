import subprocess
import os
import argparse

# Argument parser
parser = argparse.ArgumentParser(description='Creates bsub jobs to filter sequences and genes based on quality.')
parser.add_argument('script_path', help='The path to the Python script to run.')
parser.add_argument('input_dir', help='The directory containing the GFF and FASTA files.')
parser.add_argument('output_dir', help='The directory to save the filtered files.')
args = parser.parse_args()

# Ensure output directory exists
os.makedirs(args.output_dir, exist_ok=True)

# Get all GFF and FASTA files
gff_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('genome_annotations_filtered_removed_sequences.gff')])
fasta_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('genome_sequences_filtered_removed_sequences.fasta')])

# Check if there is a corresponding FASTA file for each GFF file
if len(gff_files) != len(fasta_files):
    print(len(gff_files), len(fasta_files))
    print("Mismatch between number of GFF and FASTA files.")
    exit(1)

# Run the Python script for each pair of GFF and FASTA files
for gff_file, fasta_file in zip(gff_files, fasta_files):
    # Construct the command
    command = f"python3 {args.script_path} {args.input_dir + gff_file} {args.input_dir + fasta_file} {args.output_dir}"
    print (command)
    # Create a job script
    job_script = f"""#!/bin/bash
#BSUB -J job_{gff_file}
#BSUB -o {args.output_dir}/job_{gff_file}.out
#BSUB -e {args.output_dir}/job_{gff_file}.err
#BSUB -n 1
#BSUB -M 131072
#BSUB -W 720

{command}
"""
    # Write the job script to a file
    with open(f"{args.output_dir}/job_{gff_file}.sh", "w") as f:
        f.write(job_script)

    # Submit the job script
    subprocess.run(f"bsub <  {args.output_dir}/job_{gff_file}.sh", shell = True)
