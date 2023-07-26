import argparse

# Create an argument parser
parser = argparse.ArgumentParser(description='Correct GFF file')
parser.add_argument('-i', '--input', help='Input GFF file', required=True)
parser.add_argument('-o', '--output', help='Output GFF file', required=True)

# Parse the command line arguments
args = parser.parse_args()

# Open both the input and output files
with open(args.input, 'r') as infile, open(args.output, 'w') as outfile:
    # Iterate over each line in the file
    for line in infile:
        # Skip header lines
        if line.startswith("#"):
            outfile.write(line)
        else:
            # Split the line into columns
            columns = line.strip().split('\t')
            # Check the start and end coordinates (columns 3 and 4, 0-indexed)
            if int(columns[3]) >= 0 and int(columns[4]) >= 0:
                # Write the line to the output file
                outfile.write(line)
 
