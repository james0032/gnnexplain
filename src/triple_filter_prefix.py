import torch
from typing import Dict, Tuple, List

def filter_triples_by_prefix(triples: torch.Tensor,
                            node_dict: Dict[str, int],
                            subject_prefixes: List[str],
                            object_prefixes: List[str]) -> torch.Tensor:
    """
    Filter triples to only keep those with specific subject and object prefixes.
    
    Args:
        triples: Tensor of triples (num_triples, 3) with [head, rel, tail]
        node_dict: Node to index mapping
        subject_prefixes: List of allowed prefixes for subjects (e.g., ['CHEBI', 'UNII'])
        object_prefixes: List of allowed prefixes for objects (e.g., ['MONDO'])
    
    Returns:
        Filtered triples tensor
    """
    # Create reverse mapping: index -> node_id
    idx_to_node = {v: k for k, v in node_dict.items()}
    
    filtered_indices = []
    
    print(f"\nFiltering triples with subject prefixes: {subject_prefixes}")
    print(f"and object prefixes: {object_prefixes}")
    
    for i in range(len(triples)):
        head_idx = triples[i, 0].item()
        tail_idx = triples[i, 2].item()
        
        # Get node IDs
        head_id = idx_to_node.get(head_idx, "")
        tail_id = idx_to_node.get(tail_idx, "")
        
        # Check if head starts with any subject prefix
        head_matches = any(head_id.startswith(prefix + ":") for prefix in subject_prefixes)
        
        # Check if tail starts with any object prefix
        tail_matches = any(tail_id.startswith(prefix + ":") for prefix in object_prefixes)
        
        if head_matches and tail_matches:
            filtered_indices.append(i)
    
    if len(filtered_indices) == 0:
        print(f"⚠️  Warning: No triples found matching the prefix criteria!")
        return triples  # Return all triples as fallback
    
    filtered_triples = triples[filtered_indices]
    
    print(f"✓ Filtered {len(triples)} triples down to {len(filtered_triples)} matching triples")
    print(f"  ({len(filtered_triples)/len(triples)*100:.1f}% of original)")
    
    return filtered_triples


def get_prefix_statistics(triples: torch.Tensor,
                         node_dict: Dict[str, int]) -> Dict[str, int]:
    """
    Get statistics about node prefixes in triples.
    
    Returns:
        Dictionary with prefix counts
    """
    from collections import defaultdict
    
    idx_to_node = {v: k for k, v in node_dict.items()}
    
    subject_prefixes = defaultdict(int)
    object_prefixes = defaultdict(int)
    
    for i in range(len(triples)):
        head_idx = triples[i, 0].item()
        tail_idx = triples[i, 2].item()
        
        head_id = idx_to_node.get(head_idx, "")
        tail_id = idx_to_node.get(tail_idx, "")
        
        # Extract prefix (part before first ":")
        if ":" in head_id:
            head_prefix = head_id.split(":")[0]
            subject_prefixes[head_prefix] += 1
        
        if ":" in tail_id:
            tail_prefix = tail_id.split(":")[0]
            object_prefixes[tail_prefix] += 1
    
    return {
        'subject_prefixes': dict(subject_prefixes),
        'object_prefixes': dict(object_prefixes)
    }


def print_prefix_inventory(triples: torch.Tensor,
                          node_dict: Dict[str, int],
                          name: str = "Dataset"):
    """
    Print inventory of node prefixes in the dataset.
    """
    stats = get_prefix_statistics(triples, node_dict)
    
    print(f"\n{'='*60}")
    print(f"{name} Prefix Inventory")
    print(f"{'='*60}")
    
    print(f"\n📊 Subject Prefixes (Head entities):")
    for prefix, count in sorted(stats['subject_prefixes'].items(), 
                                key=lambda x: x[1], reverse=True):
        print(f"  {prefix:20s}: {count:8,} triples")
    
    print(f"\n📊 Object Prefixes (Tail entities):")
    for prefix, count in sorted(stats['object_prefixes'].items(), 
                                key=lambda x: x[1], reverse=True):
        print(f"  {prefix:20s}: {count:8,} triples")
    
    print(f"{'='*60}\n")