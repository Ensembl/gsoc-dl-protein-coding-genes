import subprocess

# Path to your Perl script
perl_script_path = "/hps/nobackup/flicek/ensembl/genebuild/frida/scripts/high_confidence_genes_gsoc23_second_try_debugging.pl"
# Path to diamond script
diamond_script_path = "/hps/nobackup/flicek/ensembl/genebuild/frida/scripts/get_diamond_coverage.py"
# Path to protein database
protein_db = "/hps/nobackup/flicek/ensembl/genebuild/frida/data/all_mammal_proteins.dmnd"
# Output directory
output_dir = "/hps/nobackup/flicek/ensembl/genebuild/frida/test_results/"
# User
user = "ensro"

# File with connection details
file_path = "/hps/nobackup/flicek/ensembl/genebuild/frida/scripts/server_list_test.txt"

with open(file_path, "r") as file:
    for line in file:
        dbname, host, port = line.strip().split()

        # Construct the command
        command = f" perl {perl_script_path} -user {user} -host {host} -port {port} -diamond_script_path {diamond_script_path} -protein_db {protein_db} -output_dir {output_dir} -dbname {dbname}"

        # Execute the command
        subprocess.run(command, shell=True)
