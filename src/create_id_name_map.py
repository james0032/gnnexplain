"""
create_id_name_map.py - Extract ID to name mapping from nodes.jsonl

Usage:
    python create_id_name_map.py --input nodes.jsonl --output id_to_name.map
"""

import json
import argparse


def create_id_name_map(input_file: str, output_file: str):
    """
    Extract ID to name mapping from nodes.jsonl.
    
    Args:
        input_file: Path to nodes.jsonl
        output_file: Path to save id_to_name.map (TSV format)
    """
    count = 0
    
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            try:
                node = json.loads(line.strip())
                
                node_id = node.get('id', '')
                node_name = node.get('name', '')
                
                if node_id and node_name:
                    fout.write(f"{node_id}\t{node_name}\n")
                    count += 1
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line: {line[:50]}... Error: {e}")
                continue
    
    print(f"✓ Created id_to_name.map with {count} mappings")
    print(f"✓ Saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Create ID to name mapping from nodes.jsonl')
    parser.add_argument('--input', type=str, default='nodes.jsonl',
                       help='Path to nodes.jsonl file')
    parser.add_argument('--output', type=str, default='id_to_name.map',
                       help='Path to output mapping file')
    
    args = parser.parse_args()
    
    create_id_name_map(args.input, args.output)


if __name__ == '__main__':
    main()