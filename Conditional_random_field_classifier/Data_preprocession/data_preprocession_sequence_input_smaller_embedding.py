import json
import os
import argparse

VALUES_ENCODING = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0.25, 0.25, 0.25, 0.25],
    [0.5, 0, 0.5, 0],
    [0, 0.5, 0, 0.5],
    [0, 0, 0.5, 0.5],
    [0.5, 0.5, 0, 0],
    [0, 0.5, 0.5, 0],
    [0.5, 0, 0, 0.5],
    [1/3, 1/3, 0, 1/3],
    [0, 1/3, 1/3, 1/3],
    [1/3, 0, 1/3, 1/3],
    [1/3, 1/3, 1/3, 0]
]


def convert_sequence_to_4D(input_sequence):
    """
    Convert a sequence of one-hot encoded embeddings to a sequence of 4D values.

    Args:
    - input_sequence (List[List[int]]): A 2D list where each inner list is a one-hot encoded embedding.

    Returns:
    - List[List[float]]: A 2D list where each inner list is the corresponding 4D value.
    """
    output_sequence = []
    for one_hot in input_sequence:
        idx = one_hot.index(1)
        output_sequence.append(VALUES_ENCODING[idx])

    return output_sequence

def process_input_file(input_file_path, output_directory, exon_threshold):
    """
    Process the input file and write the transformed data to the output directory.
    
    Args:
    - input_file_path (str): The path to the input file.
    - output_directory (str): The directory where the output file will be saved.
    - exon_threshold (int): The threshold for the minimum number of exons.
    """
    
    output_file_name = os.path.splitext(os.path.basename(input_file_path))[0]
    output_file_path = os.path.join(output_directory, f"{output_file_name}_small_sequence_umbedding.txt")

     
    with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
        lines_skipped = 0
        lines = 0
        for line in infile:
            try:
                data_list = json.loads(line)
                exons = sum(1 for data in data_list if int(data.get('exon', 0)) == 1)
                lines += 1
                if exons <= exon_threshold:
                    lines_skipped += 1
                    continue  # Only use lines with more than 10 exons
                    

                for data in data_list:
                    features_sequence = convert_sequence_to_4D(data["sequence"])
                    features_general = [v if k not in ('repetitive') else (1-v)*10 for k, v in data.items() 
                                        if k not in ('gene', 'exon', 'position', 'sequence')]
                    position = int(data.get("position", None))
                    exon = int(data.get("exon", None))
                    gene = int(data.get("gene", None))

                    transformed_data = {
                        "features_sequence": features_sequence,
                        "features_general": features_general,
                        "exon": exon,
                        "gene": gene,
                        "position": position
                    }

                    # Save the transformed data in JSON format, one datapoint per line
                    outfile.write(json.dumps(transformed_data) + '\n')
            
            except json.JSONDecodeError as e:
                print(f"Skipping line due to error: {e}")
        print (lines, lines_skipped, lines_skipped/lines)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transform input file data.")
    parser.add_argument("input_file", help="Path to the input file.")
    parser.add_argument("output_directory", help="Directory to save the output file.")
    parser.add_argument("--exon_threshold", type=int, default=10, help="Threshold for minimum number of exons. Default is 10.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)
    
    process_input_file(args.input_file, args.output_directory, args.exon_threshold)
