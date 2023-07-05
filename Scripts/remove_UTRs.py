import sys

def filter_gff(input_file, output_file):
    # Open input and output files
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        # Iterate over each line in the input file
        for line in f_in:
            # Split the line into fields
            fields = line.strip().split('\t')
            if len(fields)>1:
                # Check if the feature is three_prime_UTR or five_prime_UTR
                if fields[2] not in ['three_prime_UTR', 'five_prime_UTR']:
                    # Write the line to the output file if it's not one of the excluded features
                    f_out.write(line)

    print("Filtering completed. Filtered GFF data saved to:", output_file)

# Check if the input file and output file are provided as command-line arguments
if len(sys.argv) < 3:
    print("Usage: python filter_gff.py <input_file> <output_file>")
else:
    # Get the input and output file paths from command-line arguments
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Call the filter_gff function with the provided file paths
    filter_gff(input_file, output_file)

 
