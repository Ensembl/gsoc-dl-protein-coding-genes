import argparse
import os
import subprocess

# Argument parser
parser = argparse.ArgumentParser(description='Executes a Python script on all GFF and FASTA file pairs in a directory.')
parser.add_argument('script_path', help='The path to the Python script to run.')
parser.add_argument('input_dir', help='The directory containing the GFF and FASTA files.')
parser.add_argument('output_dir', help='The directory to save the output files.')
args = parser.parse_args()

# Get all GFF and FASTA files
gff_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.gff')])
fasta_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.fasta')])

# Check if there is a corresponding FASTA file for each GFF file
if len(gff_files) != len(fasta_files):
    print(len(gff_files), len(fasta_files))
    print("Mismatch between number of GFF and FASTA files.")
    exit(1)

# Run the Python script for each pair of GFF and FASTA files
for gff_file, fasta_file in zip(gff_files, fasta_files):
    # Ensure the output subdirectory exists
    sub_output_dir = os.path.join(args.output_dir, os.path.splitext(gff_file)[0])
    os.makedirs(sub_output_dir, exist_ok=True)

    # Construct the command
    command = f"python3 {args.script_path} {os.path.join(args.input_dir, gff_file)} {os.path.join(args.input_dir, fasta_file)} {sub_output_dir}"

    # Print the command
    print(command)

    # Execute the command
    # Create a job script
    job_script = f"""#!/bin/bash
#BSUB -J job_{os.path.splitext(gff_file)[0]}
#BSUB -o {os.path.join(sub_output_dir, 'job.out')}
#BSUB -e {os.path.join(sub_output_dir, 'job.err')}
#BSUB -n 1
#BSUB -M 4096
#BSUB -W 120

{command}
"""

    # Write the job script to a file
    job_script_path = os.path.join(sub_output_dir, 'job.sh')
    with open(job_script_path, 'w') as f:
        f.write(job_script)

    # Make the job script executable
    os.chmod(job_script_path, 0o755)

    # Submit the job script
    os.system(f"bsub < {job_script_path}")
