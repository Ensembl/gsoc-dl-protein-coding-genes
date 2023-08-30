import os
import argparse

def submit_job(script_path, input_file, output_directory, exon_threshold):
    base_input_name = os.path.basename(input_file).split(".")[0]
    output_file = os.path.join(output_directory, f'{base_input_name}_processed_data.json')
    job_output = os.path.join(output_directory, f'{base_input_name}_job_output.txt')
    job_error = os.path.join(output_directory, f'{base_input_name}_job_error.txt')

    cmd = f'bsub -J data_preprocessing_job -M 250000 -R "rusage[mem=250000]" -o "{job_output}" -e "{job_error}" python3 "{script_path}" "{input_file}" "{output_file}" --exon_threshold {exon_threshold}'
    os.system(cmd)

def main():
    parser = argparse.ArgumentParser(description="Process txt files in an input directory.")
    parser.add_argument('input_directory', type=str, help="The directory containing the txt files to be processed.")
    parser.add_argument('output_directory', type=str, help="The directory where the processed data and logs will be saved.")
    parser.add_argument('--script_path', type=str, required=True, help="Path to the preprocessing script to execute.")
    parser.add_argument('--exon_threshold', type=int, default=10, help="The minimum number of exons required. Default is 10.")

    args = parser.parse_args()

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    for input_file in os.listdir(args.input_directory):
        if input_file.endswith(".txt"):
            submit_job(args.script_path, os.path.join(args.input_directory, input_file), args.output_directory, args.exon_threshold)


if __name__ == "__main__":
    main()
 
