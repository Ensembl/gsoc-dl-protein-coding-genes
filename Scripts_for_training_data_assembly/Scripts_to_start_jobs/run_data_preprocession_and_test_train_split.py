import os
import shutil
import random
import argparse

def submit_job(fasta_file, gff_file, output_file):
    cmd = f'bsub -J data_preprocessing_job -M 250000 -R "rusage[mem=250000]" -o job_output.txt -e job_error.txt python3 "/hps/nobackup/flicek/ensembl/genebuild/frida/scripts/gsoc-dl-protein-coding-genes/Conditional random field classifier/data_preprocession.py" "{fasta_file}" "{gff_file}" "{output_file}"'
    os.system(cmd)

def split_data_into_train_test(input_directory, output_directory, split_ratio=0.8):
    # Get list of all file pairs in directory
    file_pairs = [(f, f.replace('genome_sequences', 'genome_annotations').replace('.fasta', '.gff')) for f in os.listdir(input_directory) if f.endswith('.fasta')]

    # Randomly shuffle the list
    random.shuffle(file_pairs)

    # Calculate the number of pairs that will go into the train set
    num_train = int(len(file_pairs) * split_ratio)

    # Split the pairs into train and test sets
    train_pairs = file_pairs[:num_train]
    test_pairs = file_pairs[num_train:]

    # Create directories for train and test data
    train_directory = os.path.join(output_directory, 'training_data')
    test_directory = os.path.join(output_directory, 'test_data')
    os.makedirs(train_directory, exist_ok=True)
    os.makedirs(test_directory, exist_ok=True)

    # Copy the test pairs to the test directory and submit a job for each train pair
    for fasta_file, gff_file in test_pairs:
        shutil.copy(os.path.join(input_directory, fasta_file), os.path.join(test_directory, fasta_file))
        shutil.copy(os.path.join(input_directory, gff_file), os.path.join(test_directory, gff_file))

    for fasta_file, gff_file in train_pairs:
        output_file = os.path.join(train_directory, f'{fasta_file.split(".")[0]}_crf_training_data.txt')
        submit_job(os.path.join(input_directory, fasta_file), os.path.join(input_directory, gff_file), output_file)

def main():
    parser = argparse.ArgumentParser(description="Split fasta and gff files into train and test sets.")
    parser.add_argument('input_directory', type=str, help="The directory containing the fasta and gff files.")
    parser.add_argument('output_directory', type=str, help="The directory where the training_data and test_data directories will be created.")
    parser.add_argument('--split_ratio', type=float, default=0.8, help="The proportion of files to include in the training set. Default is 0.8.")

    args = parser.parse_args()

    split_data_into_train_test(args.input_directory, args.output_directory, args.split_ratio)


if __name__ == "__main__":
    main()
 
