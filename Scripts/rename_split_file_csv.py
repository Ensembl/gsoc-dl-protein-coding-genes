import os
import pandas as pd
import argparse
import shutil

# Setup command line argument parsing
parser = argparse.ArgumentParser(description='Update CSV files in subdirectories.')
parser.add_argument('dir', metavar='D', type=str, help='The root directory to process')
args = parser.parse_args()

root_dir = args.dir

for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename == 'fragemented_files.csv':
            # Get the subfolder name
            subfolder_name = os.path.basename(dirpath)
            
            # Create a copy of the file with new name
            new_filename = f"{subfolder_name}_fragmented_files.csv"
            shutil.copy2(os.path.join(dirpath, filename), os.path.join(dirpath, new_filename))
            
            # Load the new CSV file using pandas
            df = pd.read_csv(os.path.join(dirpath, new_filename))
            
            # Add a new column with the subfolder name
            df['subfolder_name'] = subfolder_name
            
            # Add a new column with the subfolder_filename
            df['id'] = df['subfolder_name'] + "_" + df['filename']
            
            # Save the updated dataframe to the new csv file
            df.to_csv(os.path.join(dirpath, new_filename), index=False)

print("All files have been copied and updated successfully.")
