import subprocess
import os
import argparse
import shutil

# Argument parser
parser = argparse.ArgumentParser(description='Creates bsub jobs to filter genes based on overlapping genes.')
parser.add_argument('script_path', help='The path to the Python script to run.')
parser.add_argument('input_dir', help='The directory containing the GFF files.')
parser.add_argument('output_dir', help='The directory to save the filtered files.')
args = parser.parse_args()

# Ensure output directory exists
os.makedirs(args.output_dir, exist_ok=True)

# Get all GFF files
gff_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('_annotations_filtered.gff')])

# Run the Python script for each GFF file
for gff_file in gff_files:
    # Construct the command
    command = f"python3 {args.script_path} {os.path.join(args.input_dir, gff_file)} {args.output_dir}"

    # Create a job script
    job_script = f"""#!/bin/bash
#BSUB -J job_{gff_file}
#BSUB -o {os.path.join(args.output_dir, f'job_{gff_file}.out')}
#BSUB -e {os.path.join(args.output_dir, f'job_{gff_file}.err')}
#BSUB -n 1
#BSUB -M 131072
#BSUB -W 720

{command}
"""
    # Write the job script to a file
    with open(os.path.join(args.output_dir, f"job_{gff_file}.sh"), "w") as f:
        f.write(job_script)

    # Submit the job script
    subprocess.run(f"bsub < {os.path.join(args.output_dir, f'job_{gff_file}.sh')}", shell=True)

    # Copy the corresponding FASTA file to the output directory
    fasta_file = gff_file.replace('_annotations_filtered.gff', '_sequences.fasta')
    shutil.copy(os.path.join(args.input_dir, fasta_file), os.path.join(args.output_dir, fasta_file))
