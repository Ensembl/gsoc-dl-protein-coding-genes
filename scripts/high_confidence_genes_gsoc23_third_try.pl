use warnings;
use strict;
use feature 'say';

use Bio::EnsEMBL::Translation;
use Bio::EnsEMBL::DBSQL::DBAdaptor;
use Bio::EnsEMBL::Analysis::Tools::Algorithms::ClusterUtils;
use Getopt::Long qw(:config no_ignore_case);

my $user = 'ensro';
my $dbname;
my $host;
my $port;
my $coord_system = 'toplevel';


my $diamond_script_path = "/hps/nobackup/flicek/ensembl/genebuild/frida/scripts/get_diamond_coverage.py";
my $protein_db;
my $output_dir;

my $options = GetOptions ("user|dbuser|u=s"       => \$user,
                          "host|dbhost|h=s"       => \$host,
                          "port|dbport|P=i"       => \$port,
                          "dbname|db|D=s"         => \$dbname,
                          "diamond_script_path=s" => \$diamond_script_path,
                          "protein_db=s"          => \$protein_db,
                          "output_dir=s"          => \$output_dir);

# Get the input database name
my ($input_dbname) = ($dbname =~ /([^_]+)_core/);
$input_dbname ||= $dbname;  # Use full dbname if the extraction fails
my $gff_output_file = "${output_dir}${input_dbname}_genome_annotations.gff";
my $csv_output_file = "${output_dir}${input_dbname}_confidence.csv";

# Create an output file for the Fasta annotations
my $fasta_output_file = "${output_dir}${input_dbname}_genome_sequences.fasta";
open(my $fasta_out, '>', $fasta_output_file) or die "Cannot open $fasta_output_file for writing: $!";

# Connect to the Ensembl core db
my $db = new Bio::EnsEMBL::DBSQL::DBAdaptor(
  -port    => $port,
  -user    => $user,
  -host    => $host,
  -dbname  => $dbname);

# For fetching info from the db via the API (specifically the seq regions in the genome, the genes and some metadata for the species)
my $slice_adaptor = $db->get_SliceAdaptor();
my $slices = $slice_adaptor->fetch_all('toplevel');
my $gene_adaptor = $db->get_GeneAdaptor();
my $meta_adaptor = $db->get_MetaContainer();

# Various output files
my $production_name = $meta_adaptor->get_production_name;
my $diamond_input_file_path = $output_dir."/".$production_name.".prots.fa";
my $final_output_file_path = $output_dir."/".$production_name.".confidence.csv";
my $diamond_output_file_path = $output_dir."/".$production_name.".prots.res";

# This hashref has a primary key of an Ensembl transcript stable id. Each id key then points to another
# hashref that has key value pairs that are ultimately used to grade the transcript later
my $transcript_results = {};

# Create an output file for running Diamond
open(OUT,">".$diamond_input_file_path);

# Create an output file for the GFF annotations
open(my $gff_out, '>', $gff_output_file) or die "Cannot open $gff_output_file for writing: $!";

# Create an output file for diamond
open(my $diamond_out, '>', $diamond_output_file_path) or die "Cannot open $diamond_output_file_path for writing: $!";

# Loop through each seq region in the genome (a seq region is a chromosomes or toplevel scaffold)
foreach my $slice (@$slices) {
  say $fasta_out ">".$slice->seq_region_name;
  say $fasta_out $slice->seq();
  # TESTING: uncomment to look at chromosome 4 only (note it will crash if there's no chromosome 4)
  #unless($slice->seq_region_name =~ /^\d+$/ && $slice->seq_region_name eq '4') {
  #  next;
  #}

  # Gene the genes, only use the protein coding ones and then get the representative (canonical) transcript
  # Using this transcript we get various bits of information about it to then make some decisions as
  # to whether or not it's high confidence
  # To assist making that decision, we also create a file of the associated protein sequences to run Diamond on
  # Currently the db this is run against is human and mouse representative proteins from Ensembl and all SwissProt
  # reviewed mammal proteins (note that most but not all of the human/mouse Ensembl proteins would be in the
  # SwissProt set, so some redundancy, but not an issue)
  my $genes = $slice->get_all_Genes();
  say "Processing: ".$slice->seq_region_name." (".scalar(@$genes)." genes)";
  foreach my $gene (@$genes) {
    my $biotype = $gene->biotype;
    unless($biotype eq 'protein_coding') {
      next;
    }

    my $transcript = $gene->canonical_transcript;
    unless($transcript) {
      die "Didn't find a canonical";
    }

    my $ise = $transcript->get_all_IntronSupportingEvidence();
    my $introns = $transcript->get_all_Introns();

    my $five_prime_utr = $transcript->five_prime_utr();
    my $three_prime_utr = $transcript->three_prime_utr();
    my $cds_sequence = $transcript->translateable_seq();
    my $cds_exons = $transcript->get_all_CDS();
    my $cds_introns = scalar(@$cds_exons) - 1;

    my $transcript_cds_complete = 0;
    my $transcript_five_prime_utr = 0;
    my $transcript_three_prime_utr = 0;
    my $transcript_canonical_splicing = 0;
    my $transcript_regular_intron_size = 0;
    my $transcript_intron_support = 0;

    if($cds_sequence =~ /^ATG/ and ($cds_sequence =~ /TAA$/ or $cds_sequence =~ /TAG$/ or $cds_sequence =~ /TGA$/)) {
      $transcript_cds_complete = 1;
    }

    if($transcript->five_prime_utr()) {
      $transcript_five_prime_utr = 1;
    }

    if($transcript->three_prime_utr()) {
      $transcript_three_prime_utr = 1;
    }

    # This is a cheap an potentially incorrect way of calculating CDS intron support
    # It could be incorrect if there are introns in the UTR. The issue is that the methods to work out
    # which introns are CDS introns are slow via the API. I mean actually this would be easy to implement
    # but I'm lazy and unlikely to be an actual issue
    if($cds_introns <= scalar(@$ise)) {
      $transcript_intron_support = 1;
    }

    # Run the intron check in terms of whether the introns have canonical splicing and are regularly sized
    ($transcript_canonical_splicing,$transcript_regular_intron_size) = intron_check($introns);

    # Store the info so far onto the transcript results hashref
    $transcript_results->{$transcript->stable_id()}->{'transcript_cds_complete'} = $transcript_cds_complete;
    $transcript_results->{$transcript->stable_id()}->{'transcript_five_prime_utr'} = $transcript_five_prime_utr;
    $transcript_results->{$transcript->stable_id()}->{'transcript_three_prime_utr'} = $transcript_three_prime_utr;
    $transcript_results->{$transcript->stable_id()}->{'transcript_canonical_splicing'} = $transcript_canonical_splicing;
    $transcript_results->{$transcript->stable_id()}->{'transcript_regular_intron_size'} = $transcript_regular_intron_size;
    $transcript_results->{$transcript->stable_id()}->{'transcript_intron_support'} = $transcript_intron_support;
    $transcript_results->{$transcript->stable_id()}->{'gene_info'} = $gene->stable_id()."\t".$gene->seq_region_name."\t".$gene->seq_region_start()."\t".$gene->seq_region_end();

    # Write the translation sequence to file
    my $translation = $transcript->translation;
    my $translation_seq = $translation->seq();
    say OUT ">".$transcript->stable_id();
    say OUT $translation_seq;
  }
}

close OUT;

# Run the Python script for Diamond, this script runs Diamond and generates a result file. The result file has
# a query and target coverage (based clustering hits), the error (a measure of the missing coverage on both query
# and target) and a score for how good the match was. The score considers the sequence identity (which is lowly
# weighted) and the coverage (highly weighted)
my $command = "python3 ".$diamond_script_path." $diamond_input_file_path $protein_db $diamond_output_file_path";
system($command);

# Load the output from diamond into the hashref
open(my $fh, '<', $diamond_output_file_path) or die "Could not open file '$diamond_output_file_path' $!";
while (my $row = <$fh>) {
    chomp $row;
    my ($query_id, $hit_id, $error, $perc_id, $score, $query_coverage, $target_coverage) = split "\t", $row;
    $transcript_results->{$query_id}->{'hit_id'} = $hit_id;
    $transcript_results->{$query_id}->{'error'} = $error;
    $transcript_results->{$query_id}->{'perc_id'} = $perc_id;
    $transcript_results->{$query_id}->{'score'} = $score;
    $transcript_results->{$query_id}->{'query_coverage'} = $query_coverage;
    $transcript_results->{$query_id}->{'target_coverage'} = $target_coverage;
}
close $fh;


# This is the last step, take all the info and get a confidence rating, put this in the final output file
open(OUT,">".$csv_output_file);
foreach my $transcript_id (keys %{$transcript_results}) {
    my $output_line = confidence_rating($transcript_results->{$transcript_id});
    $output_line = $dbname."\t".$host."\t".$port."\t".$output_line;
    say OUT $output_line;

    # Write gene quality annotation to the GFF output file
    my ($gene_id, $seq_region_name, $seq_region_start, $seq_region_end) = split("\t", $transcript_results->{$transcript_id}->{'gene_info'});
    my $gene_quality = $transcript_results->{$transcript_id}->{'confidence'};
    say $gff_out join("\t", $seq_region_name, 'Ensembl', 'gene_quality', $seq_region_start, $seq_region_end, '.', '.', '.', "gene_id=$gene_id;gene_quality=$gene_quality");
}
close OUT;
close $gff_out;
close $diamond_out;
close $fasta_out;
#use Data::Dumper;
#print Dumper($transcript_results);

sub confidence_rating {
  my ($transcript_result) = @_;

  # This is the method that assesses confidence. Note that there is no straightforward measure of confidence
  # as there are so many potential issues in terms of analysing the available data. For example, the intron
  # supporting evidence, which should be a good way to measure confidence, is sometimes missing. Pig, which
  # has a very good annotation and based mostly on transcriptomic data, has no intron supporting evidence
  my $confidence;
  my $transcript_cds_complete = $transcript_result->{'transcript_cds_complete'};
  my $transcript_five_prime_utr = $transcript_result->{'transcript_five_prime_utr'};
  my $transcript_three_prime_utr = $transcript_result->{'transcript_three_prime_utr'};
  my $transcript_canonical_splicing = $transcript_result->{'transcript_canonical_splicing'};
  my $transcript_regular_intron_size = $transcript_result->{'transcript_regular_intron_size'};
  my $transcript_intron_support = $transcript_result->{'transcript_intron_support'};
  my $transcript_error = $transcript_result->{'error'};
  my $transcript_score = $transcript_result->{'score'};
  my $transcript_hit_id = $transcript_result->{'hit_id'};
  my $transcript_perc_id = $transcript_result->{'perc_id'};

  # Conditions for high confidence:
  # Hard requirements: A complete CDS, the splice sites are canonical, intron sizes look regular
  # It's then a balancing act of testing different combinations of conditions. These should be tweaked
  # to test effect on the training set. At the moment it comes out on average that, a bit over half of
  # most gene sets come out as high confidence, a very small amount come out as medium confidence and
  # a little under half come out as low confidence. This implies that there's a bit of an imbalance,
  # we would expect that in reality there should be a lot more medium confidence models rather than
  # low confidence ones. It's fine to be strict in terms of the high confidence ones, but might be
  # problematic if in terms of it being so binary between high and low
  # One thing that could be done is some allowance on the non-canonical splicing, for example 95%
  # of splices are canoncial
  unless($transcript_cds_complete and defined($transcript_error)) {
    $confidence = "low";
  } elsif($transcript_regular_intron_size and $transcript_canonical_splicing and $transcript_five_prime_utr and $transcript_three_prime_utr and $transcript_error <= 10) {
    $confidence = "high";
  } elsif($transcript_regular_intron_size and $transcript_canonical_splicing and $transcript_error <= 5 and $transcript_score >= 80) {
    $confidence = "high";
  } elsif($transcript_regular_intron_size and $transcript_canonical_splicing and $transcript_error <= 10 and $transcript_intron_support) {
    $confidence = "high";
  } elsif(($transcript_regular_intron_size or $transcript_canonical_splicing) and $transcript_error <= 10 and $transcript_intron_support) {
    $confidence = "medium";
  } elsif(($transcript_regular_intron_size or $transcript_canonical_splicing) and $transcript_error <= 5 and $transcript_score >= 80) {
    $confidence = "medium";
  } else {
    $confidence = "low";
  }

  $transcript_result->{'confidence'} = $confidence;
  return($transcript_result->{'gene_info'}."\t".$transcript_result->{'confidence'});
}


sub intron_check {
  my ($introns) = @_;

  # Introns of < 50bp are quiet unusual, the machinery needs a minimum number of bases between exons
  # Introns over 100000kb are not unusual, however they're more often associated with misalignments
  my $min_length = 50;
  my $max_length = 100000;

  my $regular_size = 1;
  my $canonical = 1;

  foreach my $intron (@$introns) {
    unless($intron->is_splice_canonical()) {
      $canonical = 0;
    }

    if(($intron->length() > $max_length) or ($intron->length() < $min_length)) {
      $regular_size = 0;
    }
  }

  return($canonical,$regular_size);
}


