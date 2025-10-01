import torch
import torch.nn.functional as F
from torch.nn import Module, Embedding
from torch_geometric.nn import RGCNConv, GNNExplainer
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

'''
# ------------------------------
# 1. Example KG triples
# ------------------------------
# Entities: 0=DrugA, 1=ProteinX, 2=DiseaseB
# Relations: 0="binds", 1="treats"
triples = [
    (0, 0, 1),  # DrugA -binds-> ProteinX
    (1, 1, 2),  # ProteinX -treats-> DiseaseB
]

num_nodes = 3
num_rels = 2
'''
# Convert triples into edge_index + edge_type
edge_index = torch.tensor([[h for h, r, t in triples],
                           [t for h, r, t in triples]], dtype=torch.long)
edge_type = torch.tensor([r for h, r, t in triples], dtype=torch.long)

x = torch.arange(num_nodes)  # dummy node indices (used for embeddings)

data = Data(x=x, edge_index=edge_index, edge_type=edge_type)

# ------------------------------
# 2. R-GCN Encoder
# ------------------------------
class RGCNEncoder(Module):
    def __init__(self, num_nodes, num_rels, emb_dim=16, num_bases=None):
        super().__init__()
        self.emb = Embedding(num_nodes, emb_dim)
        self.conv1 = RGCNConv(emb_dim, emb_dim, num_rels, num_bases=num_bases)
        self.conv2 = RGCNConv(emb_dim, emb_dim, num_rels, num_bases=num_bases)

    def forward(self, x, edge_index, edge_type):
        x = self.emb(x)
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = self.conv2(x, edge_index, edge_type)
        return x  # entity embeddings

# ------------------------------
# 3. DistMult Decoder
# ------------------------------
class DistMultDecoder(Module):
    def __init__(self, num_rels, emb_dim=16):
        super().__init__()
        self.rel_emb = Embedding(num_rels, emb_dim)

    def forward(self, head, rel, tail):
        r = self.rel_emb(rel)
        return torch.sum(head * r * tail, dim=-1)  # DistMult score

# ------------------------------
# 4. Full Model = Encoder + Decoder
# ------------------------------
class KGModel(Module):
    def __init__(self, num_nodes, num_rels, emb_dim=16):
        super().__init__()
        self.encoder = RGCNEncoder(num_nodes, num_rels, emb_dim)
        self.decoder = DistMultDecoder(num_rels, emb_dim)

    def forward(self, x, edge_index, edge_type, triples):
        z = self.encoder(x, edge_index, edge_type)
        h = z[triples[:, 0]]
        r = triples[:, 1]
        t = z[triples[:, 2]]
        return self.decoder(h, r, t)

    def score_triple(self, h, r, t, x, edge_index, edge_type):
        z = self.encoder(x, edge_index, edge_type)
        return self.decoder(z[h], torch.tensor([r]), z[t])

# ------------------------------
# 5. Training loop
# ------------------------------
model = KGModel(num_nodes, num_rels, emb_dim=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training data as tensor
train_triples = torch.tensor(triples, dtype=torch.long)

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    pos_score = model(x, edge_index, edge_type, train_triples)

    # Simple negative sampling (corrupt tail)
    neg_tails = torch.randint(0, num_nodes, (len(train_triples),))
    neg_triples = torch.stack([train_triples[:,0], train_triples[:,1], neg_tails], dim=1)
    neg_score = model(x, edge_index, edge_type, neg_triples)

    loss = -torch.mean(F.logsigmoid(pos_score) + F.logsigmoid(-neg_score))
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ------------------------------
# 6. Predict a new edge
# ------------------------------
head, rel, tail = 0, 1, 2  # DrugA -treats-> DiseaseB (to explain)
score = model.score_triple(head, rel, tail, x, edge_index, edge_type)
print(f"\nPredicted score for (DrugA, treats, DiseaseB): {score.item():.4f}")

# ------------------------------
# 7. Explain with GNNExplainer
# ------------------------------
# Focus explanation on head node (0=DrugA)
explainer = GNNExplainer(model.encoder, epochs=200)

node_feat_mask, edge_mask = explainer.explain_node(
    node_idx=head,
    x=x,
    edge_index=edge_index,
    edge_type=edge_type
)

print("\nEdge importance mask:", edge_mask.detach().numpy())

# Visualize subgraph
explainer.visualize_subgraph(head, edge_index, edge_mask)