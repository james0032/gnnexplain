import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
import numpy as np
from typing import Dict, Tuple, List
import pickle
import argparse


class DistMult(nn.Module):
    """DistMult decoder for knowledge graph completion."""
    
    def __init__(self, num_relations: int, embedding_dim: int):
        super().__init__()
        self.relation_embeddings = nn.Parameter(
            torch.Tensor(num_relations, embedding_dim)
        )
        nn.init.xavier_uniform_(self.relation_embeddings)
    
    def forward(self, head_emb: torch.Tensor, tail_emb: torch.Tensor, 
                rel_idx: torch.Tensor) -> torch.Tensor:
        """
        Compute DistMult scores.
        
        Args:
            head_emb: Head entity embeddings (batch_size, embedding_dim)
            tail_emb: Tail entity embeddings (batch_size, embedding_dim)
            rel_idx: Relation indices (batch_size,)
        
        Returns:
            Scores for each triple (batch_size,)
        """
        rel_emb = self.relation_embeddings[rel_idx]
        # DistMult: <h, r, t> = sum(h * r * t)
        scores = torch.sum(head_emb * rel_emb * tail_emb, dim=1)
        return scores


class RGCNDistMultModel(nn.Module):
    """RGCN encoder with DistMult decoder for knowledge graph embedding."""
    
    def __init__(self, num_nodes: int, num_relations: int, 
                 embedding_dim: int = 128, num_layers: int = 2,
                 num_bases: int = 30, dropout: float = 0.2):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        
        # Initial node embeddings
        self.node_embeddings = nn.Parameter(
            torch.Tensor(num_nodes, embedding_dim)
        )
        nn.init.xavier_uniform_(self.node_embeddings)
        
        # RGCN layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(
                RGCNConv(embedding_dim, embedding_dim, 
                        num_relations, num_bases=num_bases)
            )
        
        self.dropout = nn.Dropout(dropout)
        self.decoder = DistMult(num_relations, embedding_dim)
    
    def encode(self, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        """
        Encode nodes using RGCN.
        
        Args:
            edge_index: Edge indices (2, num_edges)
            edge_type: Edge types (num_edges,)
        
        Returns:
            Node embeddings (num_nodes, embedding_dim)
        """
        x = self.node_embeddings
        
        for conv in self.convs:
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)
            x = self.dropout(x)
        
        return x
    
    def decode(self, node_emb: torch.Tensor, head_idx: torch.Tensor,
               tail_idx: torch.Tensor, rel_idx: torch.Tensor) -> torch.Tensor:
        """
        Decode triples using DistMult.
        
        Args:
            node_emb: Node embeddings (num_nodes, embedding_dim)
            head_idx: Head entity indices (batch_size,)
            tail_idx: Tail entity indices (batch_size,)
            rel_idx: Relation indices (batch_size,)
        
        Returns:
            Scores for each triple (batch_size,)
        """
        head_emb = node_emb[head_idx]
        tail_emb = node_emb[tail_idx]
        return self.decoder(head_emb, tail_emb, rel_idx)
    
    def forward(self, edge_index: torch.Tensor, edge_type: torch.Tensor,
                head_idx: torch.Tensor, tail_idx: torch.Tensor, 
                rel_idx: torch.Tensor) -> torch.Tensor:
        """Full forward pass."""
        node_emb = self.encode(edge_index, edge_type)
        scores = self.decode(node_emb, head_idx, tail_idx, rel_idx)
        return scores


class KGDataLoader:
    """Load and preprocess knowledge graph data."""
    
    def __init__(self, node_dict_path: str, rel_dict_path: str):
        self.node_dict = self.load_dict(node_dict_path)
        self.rel_dict = self.load_dict(rel_dict_path)
        self.num_nodes = len(self.node_dict)
        self.num_relations = len(self.rel_dict)
    
    @staticmethod
    def load_dict(path: str) -> Dict[str, int]:
        """Load entity or relation dictionary."""
        mapping = {}
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    mapping[parts[0]] = int(parts[1])
        return mapping
    
    def load_triples(self, path: str) -> torch.Tensor:
        """
        Load triples from file.
        
        Returns:
            Tensor of shape (num_triples, 3) with [head, relation, tail] indices
        """
        triples = []
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    head = self.node_dict.get(parts[0])
                    rel = self.rel_dict.get(parts[1])
                    tail = self.node_dict.get(parts[2])
                    
                    if head is not None and rel is not None and tail is not None:
                        triples.append([head, rel, tail])
        
        return torch.tensor(triples, dtype=torch.long)
    
    def create_pyg_data(self, triples: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert triples to PyG format.
        
        Returns:
            edge_index: (2, num_edges)
            edge_type: (num_edges,)
        """
        edge_index = torch.stack([triples[:, 0], triples[:, 2]], dim=0)
        edge_type = triples[:, 1]
        return edge_index, edge_type


def generate_negative_samples(positive_triples: torch.Tensor, 
                              num_nodes: int, 
                              num_negatives: int = 1) -> torch.Tensor:
    """
    Generate negative samples by corrupting head or tail entities.
    
    Args:
        positive_triples: Positive triples (num_pos, 3)
        num_nodes: Total number of nodes
        num_negatives: Number of negative samples per positive
    
    Returns:
        Negative triples (num_pos * num_negatives, 3)
    """
    num_pos = positive_triples.shape[0]
    negatives = []
    
    for _ in range(num_negatives):
        # Randomly corrupt head or tail
        corrupted = positive_triples.clone()
        corrupt_head = torch.rand(num_pos) < 0.5
        
        # Corrupt heads
        corrupted[corrupt_head, 0] = torch.randint(0, num_nodes, 
                                                    (corrupt_head.sum(),))
        # Corrupt tails
        corrupted[~corrupt_head, 2] = torch.randint(0, num_nodes, 
                                                     ((~corrupt_head).sum(),))
        negatives.append(corrupted)
    
    return torch.cat(negatives, dim=0)


def train_epoch(model: RGCNDistMultModel, 
                edge_index: torch.Tensor, 
                edge_type: torch.Tensor,
                train_triples: torch.Tensor,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                num_negatives: int = 5,
                batch_size: int = 1024) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Generate negative samples
    neg_triples = generate_negative_samples(train_triples, 
                                            model.num_nodes, 
                                            num_negatives)
    
    # Combine positive and negative samples
    all_triples = torch.cat([train_triples, neg_triples], dim=0)
    labels = torch.cat([
        torch.ones(len(train_triples)),
        torch.zeros(len(neg_triples))
    ]).to(device)
    
    # Shuffle
    perm = torch.randperm(len(all_triples))
    all_triples = all_triples[perm]
    labels = labels[perm]
    
    # Mini-batch training
    for i in range(0, len(all_triples), batch_size):
        batch_triples = all_triples[i:i+batch_size].to(device)
        batch_labels = labels[i:i+batch_size]
        
        optimizer.zero_grad()
        
        scores = model(edge_index, edge_type,
                      batch_triples[:, 0],
                      batch_triples[:, 2],
                      batch_triples[:, 1])
        
        loss = F.binary_cross_entropy_with_logits(scores, batch_labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


@torch.no_grad()
def evaluate(model: RGCNDistMultModel,
            edge_index: torch.Tensor,
            edge_type: torch.Tensor,
            test_triples: torch.Tensor,
            all_triples: torch.Tensor,
            device: torch.device,
            batch_size: int = 1024) -> Dict[str, float]:
    """
    Evaluate model on test set with MRR and Hit@10 metrics.
    
    Args:
        model: Trained model
        edge_index: Graph edge indices
        edge_type: Graph edge types
        test_triples: Test triples to evaluate
        all_triples: All triples in the dataset (for filtering)
        device: Device
        batch_size: Batch size for evaluation
    
    Returns:
        Dictionary with accuracy, MRR, and Hit@10 metrics
    """
    model.eval()
    
    # Encode all nodes once
    node_emb = model.encode(edge_index, edge_type)
    
    # Convert all_triples to a set for fast lookup
    all_triples_set = set(map(tuple, all_triples.tolist()))
    
    mrr_sum = 0.0
    hits_at_10 = 0
    num_samples = 0
    
    all_predictions = []
    all_labels = []
    
    print("Computing rankings for MRR and Hit@10...")
    
    for i in range(0, len(test_triples), batch_size):
        batch = test_triples[i:i+batch_size]
        
        for triple in batch:
            head, rel, tail = triple[0].item(), triple[1].item(), triple[2].item()
            
            # --- Evaluate tail prediction (h, r, ?) ---
            # Score all possible tails
            head_emb = node_emb[head].unsqueeze(0).expand(model.num_nodes, -1)
            tail_emb = node_emb  # All nodes as potential tails
            rel_emb = model.decoder.relation_embeddings[rel].unsqueeze(0).expand(model.num_nodes, -1)
            
            scores = torch.sum(head_emb * rel_emb * tail_emb, dim=1)
            
            # Filter out other true triples (filtered ranking)
            for j in range(model.num_nodes):
                if j != tail and (head, rel, j) in all_triples_set:
                    scores[j] = float('-inf')
            
            # Get rank of true tail
            sorted_indices = torch.argsort(scores, descending=True)
            rank = (sorted_indices == tail).nonzero(as_tuple=True)[0].item() + 1
            
            mrr_sum += 1.0 / rank
            if rank <= 10:
                hits_at_10 += 1
            num_samples += 1
            
            # For accuracy calculation
            all_predictions.append(scores[tail].item())
            all_labels.append(1.0)
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {min(i+batch_size, len(test_triples))}/{len(test_triples)} test triples")
    
    # Generate negative samples for accuracy
    neg_triples = generate_negative_samples(test_triples, model.num_nodes, num_negatives=5)
    
    for i in range(0, len(neg_triples), batch_size):
        batch_triples = neg_triples[i:i+batch_size].to(device)
        scores = model(edge_index, edge_type,
                      batch_triples[:, 0],
                      batch_triples[:, 2],
                      batch_triples[:, 1])
        all_predictions.extend(scores.cpu().tolist())
        all_labels.extend([0.0] * len(scores))
    
    # Calculate accuracy
    all_predictions = torch.tensor(all_predictions)
    all_labels = torch.tensor(all_labels)
    predictions = (torch.sigmoid(all_predictions) > 0.5).float()
    accuracy = (predictions == all_labels).float().mean().item()
    
    # Calculate final metrics
    mrr = mrr_sum / num_samples
    hit_at_10 = hits_at_10 / num_samples
    
    return {
        'accuracy': accuracy,
        'mrr': mrr,
        'hit@10': hit_at_10
    }


def explain_triples(model: RGCNDistMultModel,
                   edge_index: torch.Tensor,
                   edge_type: torch.Tensor,
                   test_triples: torch.Tensor,
                   device: torch.device,
                   num_samples: int = 10) -> List[Dict]:
    """
    Use GNNExplainer to explain test triples.
    
    Args:
        model: Trained model
        edge_index: Graph edge indices
        edge_type: Graph edge types
        test_triples: Test triples to explain
        device: Device
        num_samples: Number of test samples to explain
    
    Returns:
        List of explanations
    """
    model.eval()
    
    # Create a wrapper for GNNExplainer
    class ModelWrapper(nn.Module):
        def __init__(self, base_model, edge_index, edge_type, target_triple):
            super().__init__()
            self.base_model = base_model
            self.edge_index = edge_index
            self.edge_type = edge_type
            self.target_triple = target_triple
        
        def forward(self, x, edge_index, edge_attr=None):
            # Encode with potentially masked edges
            node_emb = self.base_model.encode(edge_index, edge_attr)
            # Decode target triple
            score = self.base_model.decode(
                node_emb,
                self.target_triple[0:1],
                self.target_triple[2:3],
                self.target_triple[1:2]
            )
            return score
    
    explanations = []
    
    # Sample test triples to explain
    sample_indices = torch.randperm(len(test_triples))[:num_samples]
    
    for idx in sample_indices:
        triple = test_triples[idx].to(device)
        
        # Create wrapper
        wrapper = ModelWrapper(model, edge_index, edge_type, triple).to(device)
        
        # Create explainer
        explainer = Explainer(
            model=wrapper,
            algorithm=GNNExplainer(epochs=200),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='regression',
                task_level='graph',
                return_type='raw',
            ),
        )
        
        # Generate explanation
        try:
            data = Data(
                x=model.node_embeddings.detach(),
                edge_index=edge_index,
                edge_attr=edge_type
            ).to(device)
            
            explanation = explainer(
                x=data.x,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr
            )
            
            explanations.append({
                'triple': triple.cpu().tolist(),
                'edge_mask': explanation.edge_mask.cpu() if hasattr(explanation, 'edge_mask') else None,
                'node_mask': explanation.node_mask.cpu() if hasattr(explanation, 'node_mask') else None,
            })
        except Exception as e:
            print(f"Error explaining triple {triple.cpu().tolist()}: {str(e)}")
    
    return explanations


def main():
    """Main training and explanation pipeline."""
    # Argument parser
    parser = argparse.ArgumentParser(description='Knowledge Graph RGCN-DistMult Training')
    
    # Model hyperparameters
    parser.add_argument('--embedding_dim', type=int, default=128,
                       help='Dimension of entity and relation embeddings')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of RGCN layers')
    parser.add_argument('--num_bases', type=int, default=30,
                       help='Number of bases for RGCN')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    
    # Training hyperparameters
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1024,
                       help='Batch size for training')
    parser.add_argument('--num_negatives', type=int, default=5,
                       help='Number of negative samples per positive sample')
    parser.add_argument('--val_frequency', type=int, default=10,
                       help='Frequency (in epochs) to run validation')
    
    # Data paths
    parser.add_argument('--node_dict', type=str, default='node_dict.txt',
                       help='Path to node dictionary file')
    parser.add_argument('--rel_dict', type=str, default='rel_dict.txt',
                       help='Path to relation dictionary file')
    parser.add_argument('--train_file', type=str, default='robo_train.txt',
                       help='Path to training triples file')
    parser.add_argument('--val_file', type=str, default='robo_val.txt',
                       help='Path to validation triples file')
    parser.add_argument('--test_file', type=str, default='robo_test.txt',
                       help='Path to test triples file')
    
    # Explanation parameters
    parser.add_argument('--num_explain', type=int, default=10,
                       help='Number of test triples to explain')
    parser.add_argument('--skip_explanation', action='store_true',
                       help='Skip explanation generation')
    
    # Output paths
    parser.add_argument('--model_save_path', type=str, default='best_model.pt',
                       help='Path to save best model')
    parser.add_argument('--explanation_save_path', type=str, default='explanations.pkl',
                       help='Path to save explanations')
    
    args = parser.parse_args()
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"\nHyperparameters:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    
    print("\n" + "="*50)
    print("Loading data...")
    print("="*50)
    data_loader = KGDataLoader(args.node_dict, args.rel_dict)
    
    train_triples = data_loader.load_triples(args.train_file)
    val_triples = data_loader.load_triples(args.val_file)
    test_triples = data_loader.load_triples(args.test_file)
    
    print(f"Loaded {len(train_triples)} train, {len(val_triples)} val, {len(test_triples)} test triples")
    print(f"Num nodes: {data_loader.num_nodes}, Num relations: {data_loader.num_relations}")
    
    # All triples for filtered ranking
    all_triples = torch.cat([train_triples, val_triples, test_triples], dim=0)
    
    # Create graph structure from training data
    edge_index, edge_type = data_loader.create_pyg_data(train_triples)
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)
    
    print("\n" + "="*50)
    print("Initializing model...")
    print("="*50)
    model = RGCNDistMultModel(
        num_nodes=data_loader.num_nodes,
        num_relations=data_loader.num_relations,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        num_bases=args.num_bases,
        dropout=args.dropout
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    print("\n" + "="*50)
    print("Training...")
    print("="*50)
    best_val_mrr = 0
    for epoch in range(args.num_epochs):
        loss = train_epoch(model, edge_index, edge_type, train_triples,
                          optimizer, device, num_negatives=args.num_negatives,
                          batch_size=args.batch_size)
        
        print(f"Epoch {epoch+1}/{args.num_epochs}, Loss: {loss:.4f}")
        
        if (epoch + 1) % args.val_frequency == 0:
            print(f"\n--- Validation at Epoch {epoch+1} ---")
            val_metrics = evaluate(model, edge_index, edge_type, 
                                  val_triples, all_triples, device,
                                  batch_size=args.batch_size)
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"Val MRR: {val_metrics['mrr']:.4f}")
            print(f"Val Hit@10: {val_metrics['hit@10']:.4f}")
            print("-" * 35 + "\n")
            
            if val_metrics['mrr'] > best_val_mrr:
                best_val_mrr = val_metrics['mrr']
                torch.save(model.state_dict(), args.model_save_path)
                print(f"✓ Saved best model with MRR: {best_val_mrr:.4f}\n")
    
    print("\n" + "="*50)
    print("Evaluating on test set...")
    print("="*50)
    model.load_state_dict(torch.load(args.model_save_path))
    test_metrics = evaluate(model, edge_index, edge_type, test_triples, 
                           all_triples, device, batch_size=args.batch_size)
    
    print("\n" + "="*50)
    print("FINAL TEST RESULTS")
    print("="*50)
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test MRR:      {test_metrics['mrr']:.4f}")
    print(f"Test Hit@10:   {test_metrics['hit@10']:.4f}")
    print("="*50)
    
    if not args.skip_explanation:
        print("\n" + "="*50)
        print("Generating explanations...")
        print("="*50)
        explanations = explain_triples(model, edge_index, edge_type, 
                                       test_triples, device, 
                                       num_samples=args.num_explain)
        
        # Save explanations
        with open(args.explanation_save_path, 'wb') as f:
            pickle.dump(explanations, f)
        
        print(f"Generated {len(explanations)} explanations")
        print(f"Explanations saved to {args.explanation_save_path}")
    
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)


if __name__ == '__main__':
    main()