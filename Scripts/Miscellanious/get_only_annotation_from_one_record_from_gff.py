
import argparse

def extract_annotations(gff_file, record_id):
    annotations = []

    with open(gff_file, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue

            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue

            seq_id = fields[0]
            source = fields[1]
            feature = fields[2]
            start = int(fields[3])
            end = int(fields[4])
            strand = fields[6]
            attributes = fields[8]

            if seq_id == record_id:
                annotations.append('\t'.join(fields) + '\n')

    return annotations

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract annotations for a specific record from a GFF file.')
    parser.add_argument('gff_file', help='Path to the GFF file')
    parser.add_argument('record_id', help='ID of the specific record to extract annotations for')
    args = parser.parse_args()

    annotations_for_record = extract_annotations(args.gff_file, args.record_id)

    output_file = f'{args.record_id}_annotations.gff'
    with open(output_file, 'w') as out_file:
        out_file.write(''.join(annotations_for_record))

    print(f"Annotations for record ID '{args.record_id}' have been written to {output_file}.")
