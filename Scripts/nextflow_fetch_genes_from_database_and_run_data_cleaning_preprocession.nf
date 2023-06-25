#!/usr/bin/env nextflow

params.output_dir = "/hps/nobackup/flicek/ensembl/genebuild/frida/results_with_full_gff_and_repeat_annotation_separated_by_strand/unfiltered_genomes/"

params.output_dir_filtered_annotations = "/hps/nobackup/flicek/ensembl/genebuild/frida/results_with_full_gff_and_repeat_annotation/filtered_annotations/"

params.output_dir_filtered_genomes = "/hps/nobackup/flicek/ensembl/genebuild/frida/results_with_full_gff_and_repeat_annotation/filtered_genomes/"

params.output_dir_splitted_files_genomes = "/hps/nobackup/flicek/ensembl/genebuild/frida/results_with_full_gff_and_repeat_annotation/splitted_files_filtered_genomes/"

params.output_dir_splitted_files_annotations = "/hps/nobackup/flicek/ensembl/genebuild/frida/results_with_full_gff_and_repeat_annotation/splitted_files_filtered_annotations/"

params.err_dir = "/hps/nobackup/flicek/ensembl/genebuild/frida/errors_output/"

params.server_list = "/hps/nobackup/flicek/ensembl/genebuild/frida/scripts/server_list.txt"

Channel
    .fromPath(params.server_list)
    .splitCsv(header:false, sep:' ', strip:true)
    .map { it -> tuple(it[0], it[1], it[2]) }
    .set { servers }

process run_perl_script {
    publishDir params.output_dir, mode: 'copy', pattern: "*.{fasta,gff}"

    output:
    tuple val(dbname), path("${dbname}_genome_sequences.fasta"), path("${dbname}_genome_annotations.gff") into perled_files

    input:
    tuple val(dbname), val(host), val(port) from servers

    script:
    """
    perl /hps/nobackup/flicek/ensembl/genebuild/frida/scripts/high_confidence_genes_gsoc23_full_gff.pl -user ensro -host ${host} -port ${port} -diamond_script_path /hps/nobackup/flicek/ensembl/genebuild/frida/scripts/get_diamond_coverage.py -protein_db /hps/nobackup/flicek/ensembl/genebuild/frida/data/all_mammal_proteins.dmnd -output_dir ${params.output_dir} -dbname ${dbname}
    """
}

process run_removal_of_overlapping_genes_annotation {
    publishDir params.output_dir_filtered_annotations, mode: 'copy'

    output:
    tuple val(dbname), path("${dbname}_genome_annotations_filtered.gff") into filtered_annotations

    input:
    tuple val(dbname), path(gff_file) from perled_files

    script:
    """
    python3 /hps/nobackup/flicek/ensembl/genebuild/frida/scripts/filter_gff_and_overlapping.py ${gff_file} ""
    """
}

process run_removal_of_overlapping_genes{
    publishDir params.output_dir_filtered_genomes, mode: 'copy'

    output:
    tuple val(dbname), path("${dbname}_filtered_removed_sequences.fasta"), path("${dbname}_filtered_removed_sequences.gff") into filtered_genomes

    input:
    tuple val(dbname), path(fasta_file), path(gff_file) from perled_files

    script:
    """
    python3 /hps/nobackup/flicek/ensembl/genebuild/frida/scripts/remove_low_quality_and_overlapping_genes.py  ${gff_file} ${fasta_file} ""
    """
}

process run_split_gff_and_fasta_to_genbank_on_annotations {
    publishDir params.output_dir_splitted_files_annotations, mode: 'copy', pattern: "*.{png,gbk,csv}"

    output:
    file('*.{png,gbk,csv}') into splitted_files_annotations

    input:
    tuple val(dbname), path(gff_file) from filtered_annotations
    path fasta_file from perled_files[1] // Using index to get fasta_file from the perled_files tuple

    script:
    """
    python3 /hps/nobackup/flicek/ensembl/genebuild/frida/scripts/split_gff_and_fasta_to_genbank.py ${gff_file} ${fasta_file} ""
    """
}

process run_split_gff_and_fasta_to_genbank_on_genomes {
    publishDir params.output_dir_splitted_files_genomes, mode: 'copy', pattern: "*.{png,gbk,csv}"

    output:
    file('*.{png,gbk,csv}') into splitted_files_genomes

    input:
    tuple val(dbname), path(fasta_file), path(gff_file) from filtered_genomes

    script:
    """
    python3 /hps/nobackup/flicek/ensembl/genebuild/frida/scripts/split_gff_and_fasta_to_genbank.py ${gff_file} ${fasta_file} ""
    """
}
