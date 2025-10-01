import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
import numpy as np
from typing import Dict, Tuple, List
import pickle


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
            device: torch.device,
            batch_size: int = 1024) -> Dict[str, float]:
    """Evaluate model on test set."""
    model.eval()
    
    # Generate negative samples for evaluation
    neg_triples = generate_negative_samples(test_triples, 
                                            model.num_nodes, 
                                            num_negatives=10)
    
    all_triples = torch.cat([test_triples, neg_triples], dim=0)
    labels = torch.cat([
        torch.ones(len(test_triples)),
        torch.zeros(len(neg_triples))
    ])
    
    all_scores = []
    for i in range(0, len(all_triples), batch_size):
        batch_triples = all_triples[i:i+batch_size].to(device)
        scores = model(edge_index, edge_type,
                      batch_triples[:, 0],
                      batch_triples[:, 2],
                      batch_triples[:, 1])
        all_scores.append(scores.cpu())
    
    all_scores = torch.cat(all_scores)
    predictions = (torch.sigmoid(all_scores) > 0.5).float()
    
    accuracy = (predictions == labels).float().mean().item()
    
    return {'accuracy': accuracy}


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
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding_dim = 128
    num_layers = 2
    num_bases = 30
    learning_rate = 0.01
    num_epochs = 100
    batch_size = 1024
    
    print("Loading data...")
    data_loader = KGDataLoader('node_dict.txt', 'rel_dict.txt')
    
    train_triples = data_loader.load_triples('robo_train.txt')
    val_triples = data_loader.load_triples('robo_val.txt')
    test_triples = data_loader.load_triples('robo_test.txt')
    
    print(f"Loaded {len(train_triples)} train, {len(val_triples)} val, {len(test_triples)} test triples")
    print(f"Num nodes: {data_loader.num_nodes}, Num relations: {data_loader.num_relations}")
    
    # Create graph structure from training data
    edge_index, edge_type = data_loader.create_pyg_data(train_triples)
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)
    
    print("Initializing model...")
    model = RGCNDistMultModel(
        num_nodes=data_loader.num_nodes,
        num_relations=data_loader.num_relations,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_bases=num_bases
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Training...")
    best_val_acc = 0
    for epoch in range(num_epochs):
        loss = train_epoch(model, edge_index, edge_type, train_triples,
                          optimizer, device, batch_size=batch_size)
        
        if (epoch + 1) % 10 == 0:
            val_metrics = evaluate(model, edge_index, edge_type, 
                                  val_triples, device)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, "
                  f"Val Acc: {val_metrics['accuracy']:.4f}")
            
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                torch.save(model.state_dict(), 'best_model.pt')
    
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load('best_model.pt'))
    test_metrics = evaluate(model, edge_index, edge_type, test_triples, device)
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    
    print("\nGenerating explanations...")
    explanations = explain_triples(model, edge_index, edge_type, 
                                   test_triples, device, num_samples=10)
    
    # Save explanations
    with open('explanations.pkl', 'wb') as f:
        pickle.dump(explanations, f)
    
    print(f"Generated {len(explanations)} explanations")
    print("Explanations saved to explanations.pkl")


if __name__ == '__main__':
    main()