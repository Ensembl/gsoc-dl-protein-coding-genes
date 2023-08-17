import subprocess

# Path to your Perl script
perl_script_path = "/hps/nobackup/flicek/ensembl/genebuild/frida/scripts/high_confidence_genes_gsoc23_full_gff_with_repeats_with_strand.pl"
# Path to diamond script
diamond_script_path = "/hps/nobackup/flicek/ensembl/genebuild/frida/scripts/get_diamond_coverage.py"
# Path to protein database
protein_db = "/hps/nobackup/flicek/ensembl/genebuild/frida/data/all_mammal_proteins.dmnd"
# Output directory
output_dir = "/hps/nobackup/flicek/ensembl/genebuild/frida/results_with_full_gff_and_repeat_annotation_separated_by_strand/"
# User
user = "ensro"

# File with connection details
file_path = "/hps/nobackup/flicek/ensembl/genebuild/frida/scripts/server_list.txt"

with open(file_path, "r") as file:
    for line in file:
        dbname, host, port = line.strip().split()

        # Construct the command
        command = f" perl {perl_script_path} -user {user} -host {host} -port {port} -diamond_script_path {diamond_script_path} -protein_db {protein_db} -output_dir {output_dir} -dbname {dbname}"

        # Create a job script
        job_script = f"""#!/bin/bash
#BSUB -J job_{dbname}
#BSUB -o {output_dir}/job_{dbname}.out
#BSUB -e {output_dir}/job_{dbname}.err
#BSUB -n 1
#BSUB -M 250072
#BSUB -W 720

{command}
"""

        # Write the job script to a file
        with open(f"{output_dir}/job_{dbname}.sh", "w") as f:
            f.write(job_script)

        # Submit the job script
        subprocess.run(f"bsub < {output_dir}/job_{dbname}.sh", shell=True)
