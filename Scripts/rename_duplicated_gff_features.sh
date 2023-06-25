#!/bin/bash

DIR=$1  # Directory passed as argument

# Move into the directory
cd "$DIR"

# Loop over all .gff files
for FILE in *.gff
do
  # Rename the file
  mv "$FILE" "${FILE%.gff}_with_duplicates.gff"

  # Create a script for each file
  echo "#!/bin/bash
  #BSUB -o ${FILE%.gff}_job_output.%J.out # specify output file
  #BSUB -e ${FILE%.gff}_job_output.%J.err # specify error file
  
  python3 -c \"
import re

def rename_duplicate_ids(filename):
    id_count = {}
    output = []

    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('#'):
                output.append(line.rstrip())
                continue
            
            columns = line.rstrip().split('\t')
            if len(columns) >= 9:  # Check if the line has at least 9 columns
                attributes = columns[8]
                if len(attributes.split(';')) >= 1:  # Check if the attributes have at least one item
                    match = re.search('ID=([^;]*);', attributes)
                    if match:
                        id = match.group(1)
                        if id in id_count:
                            id_count[id] += 1
                            new_id = f'{id}_{id_count[id]}'
                            new_attributes = attributes.replace(f'ID={id};', f'ID={new_id};')
                            columns[8] = new_attributes
                    else:
                        id_count[id] = 0
                output.append('\t'.join(columns))

    with open(filename.replace('_with_duplicates', ''), 'w') as file:
        for line in output:
            file.write(line + '\n')

rename_duplicate_ids('${FILE%.gff}_with_duplicates.gff')
\" 
" > "${FILE%.gff}_job_script.sh"
  
  # Make the script executable
  chmod +x "${FILE%.gff}_job_script.sh"

  # Submit the script as a job
  bsub < "${FILE%.gff}_job_script.sh"
done
