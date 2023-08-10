import os
import argparse


def submit_job(file_path, output_directory):
    # Extract the filename without extension
    filename = os.path.splitext(os.path.basename(file_path))[0]

    # Generate error and output filenames
    error_filename = os.path.join(
        output_directory, f'{filename}_job_error.txt')
    output_filename = os.path.join(
        output_directory, f'{filename}_job_output.txt')

    # Generate command to submit the job
    cmd = f'bsub -J data_preprocessing_job -M 250000 -R "rusage[mem=250000]" -o "{output_filename}" -e "{error_filename}" python3 "/hps/nobackup/flicek/ensembl/genebuild/frida/scripts/gsoc-dl-protein-coding-genes/Conditional random field classifier/crf_classifier.py" -f "{file_path}" -o "{output_directory}"'
    os.system(cmd)


if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(
        description="Submit jobs to train a CRF classifier on genomic data.")

    # Add arguments
    parser.add_argument('-d', '--directory', required=True,
                        type=str, help="Directory containing .txt files")
    parser.add_argument('-o', '--output_directory', required=True,
                        type=str, help="Output directory for results")

    # Parse arguments
    args = parser.parse_args()

    # Get all .txt files in the directory
    txt_files = [os.path.join(args.directory, f) for f in os.listdir(
        args.directory) if f.endswith('.txt')]

    # Submit a job for each .txt file
    for file in txt_files:
        submit_job(file, args.output_directory)
