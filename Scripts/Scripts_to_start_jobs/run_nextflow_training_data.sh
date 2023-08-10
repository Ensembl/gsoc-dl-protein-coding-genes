#!/bin/bash

module load nextflow-20.07.1-gcc-9.3.0-mqfchke  # assuming that you are using environment modules
nextflow run nextflow_fetch_genes_from_database_and_run_data_cleaning_preprocession.nf
