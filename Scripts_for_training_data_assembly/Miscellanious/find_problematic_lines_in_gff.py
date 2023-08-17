import gffutils
import tempfile
import os
import argparse

# Create an argument parser
parser = argparse.ArgumentParser(description='Screen GFF file')
parser.add_argument('-g', '--gff', help='Input GFF file', required=True)

# Parse the command line arguments
args = parser.parse_args()

# Open the input file
with open(args.gff, 'r') as infile:
    # Iterate over each line in the file, keeping track of line numbers
    for line_number, line in enumerate(infile, 1):
            # Create a temporary file
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp:
                # Write the current line to the temporary file
                temp.write(line)
                temp_name = temp.name
            if line_number%1_000_000 == 0:
                try:
                    # Try to create a database from the temporary file
                    gffutils.create_db(temp_name, dbfn=':memory:', force=True, keep_order=True,
                                       merge_strategy='merge', sort_attribute_values=True)
                    print(line_number)
                except Exception as e:
                    # If an error occurs, print the line number and the error
                    print(f"Error on line {temp}: {e}")

                # Delete the temporary file
    os.remove(temp_name)
