import subprocess
import os
import argparse

# Argument parser
parser = argparse.ArgumentParser(description='Creates bsub jobs to filter bad features.')
parser.add_argument('script_path', help='The path to the Python script to run.')
parser.add_argument('input_dir', help='The directory containing the GFF and FASTA files.')
args = parser.parse_args()

# Get all GFF and FASTA files
gff_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.gff')])

# Run the Python script for each pair of GFF and FASTA files
for gff_file in gff_files:
    # Construct the command
    command = f"python3 {args.script_path} -i {args.input_dir + gff_file} -o {args.input_dir + gff_file.replace('gff', 'removed_negative_features.gff')}"
    print (command)
    # Create a job script
    job_script = f"""#!/bin/bash
#BSUB -J job_{gff_file}
#BSUB -o {args.input_dir}/job_remove_negative_features_{gff_file}.out
#BSUB -e {args.input_dir}/job_remove_negative_features_{gff_file}.err
#BSUB -n 1
#BSUB -M 25072
#BSUB -W 720

{command}
"""
    # Write the job script to a file
    with open(f"{args.input_dir}/job_{gff_file}.sh", "w") as f:
        f.write(job_script)

    # Submit the job script
    subprocess.run(f"bsub <  {args.input_dir}/job_{gff_file}.sh", shell = True)
