import os
import sys
import subprocess
from collections import defaultdict
from Bio import SeqIO

# Get the fasta file, the database, and the output file from the command-line arguments
fasta_file = sys.argv[1]
database = sys.argv[2]
output_file = sys.argv[3]
print_detailed = 0
if len(sys.argv) > 4:
    print_detailed = 1

# Parse the fasta file to get the length of each query sequence
query_lengths = {seq.id: len(seq) for seq in SeqIO.parse(fasta_file, "fasta")}

# Run the diamond blastp command
# Note that nident, length and gaps are used to calculate the ungapped sequence similarity
# Masking is set at 0 to make the sequence similarity calculation more correct
# There's an assumption here that the best hit for each target is full length, Diamond is
# fairly generous in terms of gaps, so if the hit was broken it should probably be low confidence
# It could be updated to later on handle multiple hits, but probably needlessly complex
# To do this you'd want the --max-hsps flag
command = f"diamond blastp -d {database} -q {fasta_file} --masking 0 -f 6 qseqid sseqid qstart qend slen pident nident length gaps -o {output_file}"
subprocess.run(command, shell=True)

# Parse the output from diamond
results = defaultdict(list)
with open(output_file) as f:
    for line in f:
        qseqid, sseqid, qstart, qend, slen, pident, nident, length, gaps = line.strip().split()
        qstart, qend, slen, pident, nident, length, gaps = map(float, [qstart, qend, slen, pident, nident, length, gaps])
        results[qseqid].append((sseqid, qstart, qend, slen, pident, nident, length, gaps))

# Compute the error term for each pair of query and target sequences
best_matches = {}
best_scores = {}
for qseqid, hits in results.items():
    best_match = None
    best_score = float("-inf")
    for sseqid, qstart, qend, slen, pident, nident, length, gaps in hits:
        query_covered = qend - qstart + 1
        target_covered = qend - qstart + 1
        query_coverage = query_covered / query_lengths[qseqid] * 100
        target_coverage = target_covered / slen * 100
        error_term = abs(query_coverage - 100) + abs(target_coverage - 100)
        # Calculate a score as a weighted average of sequence similarity and error term
        pident_adj = nident / (length - gaps) * 100
        score = 0.25 * pident_adj + 0.75 * (100 - error_term)
        if score > best_score:
            best_score = score
            best_match = (sseqid, pident_adj, error_term, query_coverage, target_coverage)
    best_matches[qseqid] = best_match
    best_scores[qseqid] = best_score

# Write the best match, error, identity, and corresponding score for each query sequence to the output file
with open(output_file, "w") as f:
    for qseqid, (sseqid, pident_adj, error_term, query_coverage, target_coverage) in best_matches.items():
        f.write(f"{qseqid}\t{sseqid}\t{error_term:.2f}\t{pident_adj:.2f}\t{best_scores[qseqid]:.2f}\t{query_coverage:.2f}\t{target_coverage:.2f}\n")

# Print the best match and corresponding score for each query sequence
if print_detailed:
  for qseqid, (sseqid, pident_adj, error_term, query_coverage, target_coverage) in best_matches.items():
      print(f"Query sequence {qseqid}, best match {sseqid}:")
      print(f"  Sequence similarity: {pident_adj}%")
      print(f"  Query coverage: {query_coverage:.2f}%")
      print(f"  Target coverage: {target_coverage:.2f}%")
      print(f"  Error term: {error_term:.2f}%")
      print(f"  Score: {best_scores[qseqid]:.2f}")
