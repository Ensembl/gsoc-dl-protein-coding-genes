import subprocess

# Path to your Perl script
perl_script_path = "/hps/software/users/ensembl/repositories/fergal/ensembl-common/scripts/high_confidence_genes_gsoc23.pl"
# Path to diamond script
diamond_script_path = "/hps/software/users/ensembl/repositories/fergal/ensembl-common/scripts/get_diamond_coverage.py"
# Path to protein database
protein_db = "/nfs/production/flicek/ensembl/genebuild/fergal/ml_progression/gsoc2023/all_mammal_proteins.dmnd"
# Output directory
output_dir = "/nfs/production/flicek/ensembl/genebuild/fergal/ml_progression/gsoc2023/results/"
# User
user = "ensro"

# File with connection details
file_path = "/nfs/production/flicek/ensembl/genebuild/fergal/ml_progression/gsoc2023/server_list.txt"

with open(file_path, "r") as file:
    for line in file:
        dbname, host, port = line.strip().split()

        # Construct the command
        command = f"bs \'perl {perl_script_path} -user {user} -host {host} -port {port} -diamond_script_path {diamond_script_path} -protein_db {protein_db} -output_dir {output_dir} -dbname {dbname}\' {dbname} 5000"

        # Execute the command
        subprocess.run(command, shell=True)
