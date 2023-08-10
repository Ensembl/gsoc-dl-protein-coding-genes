# File Structure Overview

This document describes the file structure of the project, organized into two main directories, each containing five subdirectories.

## Directory 1: `Scripts`

### Description

`Scripts` is responsible for fetching of training data from the ensemble API, evaluating the quality of gene annotations and then data cleaning.

### Subdirectories

#### `Ensembl_api_training_data_download_and_gene_confidence_calculation`

- **Description**: Main directory for fetching Genomes from the Ensembl api and evaluating the gene annotation quality.
- **Files**:
  - `server_list.txt`: Server list for all mammal genomes.
  - `high_confidence_genes_gsoc23*.pl`: Fetches genomes from the perl api, runs "get_diamond_coverage.py" and evaluates the quality of their gene annotations based on intron size, intron support, canonical splicing and sequence identity to known protein/coverage. It saves a fasta file containing the genomic sequences and a gff file containing, genes, repeat features, UTRs, introns and exons. Additionally, it saves a CSV file including the gene quality.
    - Most up to date version: `high_confidence_genes_gsoc23_full_gff_with_repeats_with_strand.pl`
  - `get_diamond_coverage.py`: Get the diamond coverage of a gene against a reference database of human genes.

#### `Training_data_cleaning`

- **Description**: Data cleaning for obtaining genomes with only high quality gene annotations.
- **Files**:
  - `filter_gff*`: Filters only the gff files -> sequence of regions with lower confidence gene annotations remain the same.
  - `remove_*_genes.py`: Filter gff and excise sequence of lower confidence from fasta file.
  - `*overlapping*`: Filter out overlapping genes.
  - `*low_confidence*`: Filter low confidence genes.
  - `correct_gff_files.py`: Removes invalid features from gff files.

#### `Scripts_to_start_jobs`

- **Description**: Python or shell scripts submitting jobs, but no higher logic code.

#### `Nextflow_data_preprocession`

- **Description**: A nextflow pipeline that implements training data aquisition and cleaning.


#### `Miscellanious`

- **Description**: Random scripts, mainly used for debugging and runnning minimal datasets locally.


## Directory 2: `Conditional random field classifier`

### Description

`Conditional random field classifier` is responsible for implementing the first phase of prediciton, the coarse grained feature extraction.

### Subdirectories

#### `Sequence classification`

- **Description**: This directory is responsible for the implementation of the neural network for prediciting the exon status of 250 bp tokens.
- **Files**:
  - `*crf*`: Implementation of conditional random field classifiers.
  - `*lstm*`: Implementation of long-short-term-memory-neural-networks.
  - `shallow-learning`: Debugging implemntation of SGDClassifiers.
  Most up to date version:
    `lstm_classifier_pytorch_one_gpu_with_line_shuffling.py`

##### `runs`

- **Description**: Data on all runs for TorchBoard.

#### `Region_concatenating`

- **Description**: Concatenates tokens with predicted exon status to larger exon-rich regions.
- **Files**:
  - `sequence_classification.py`: Concatenates tokens with predicted exon status to larger exon-rich regions.
  - `plotting_results_with_region_detection.py`: Methods for plotting the results of this region finding.

#### `Data_proprocession`

- **Description**: This directory is responsible for feature extraction - cutting the sequences up in 250 bp sniplets, calculating their k-mer content and determining the repeat and exon status.
- **Files**:
  - `data_preprocession.py`: Cutting the sequences up in 250 bp sniplets, calculating their k-mer content and determining the repeat and exon status
  - `data_preprocession_with_size_redistribution.py`: Most up to date version, outputs a txt file with each line representing 4000*250 bp

#### `Debugging`

- **Description**: Small scripts used to create artificial test data.
