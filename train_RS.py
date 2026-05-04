import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.gcn import GCNEncoder

# Dataset Load
dataset = Planetoid(root='data', name='Cora')
data = dataset[0]

print(f"Data Load Completed.")
print(f"Nodes (Paper) : {data.num_nodes}")
print(f"Edges (Citation) : {data.num_edges}")
print(f"Classes : {dataset.num_classes}")
print(f"Feature Dimension : {dataset.num_node_features}")

# Define Title of Papers
CLASS_NAMES = [
    "Case Based",
    "Genetic Algorithms",
    "Neural Networks",
    "Probabilistic Methods",
    "Reinforce Learning",
    "Rule Learning",
    "Theory"
]

def make_title(node_id, label):
    prefixes = [
        "A Study on", "Learning with", "Advances in",
        "An Approach to", "Analysis of", "Efficient",
        "Robust", "Deep", "Scalable", "Bayesian"
    ]
    prefix = prefixes[node_id % len(prefixes)]
    return f"{prefix} {CLASS_NAMES[label]} (Paper #{node_id})"

paper_titles = [
    make_title(i, data.y[i].item()) for i in range(data.num_nodes)
]

# Train
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nTraining Device : {device}")

model = GCNEncoder(
    in_channels=dataset.num_node_features,
    hidden_channels = 128,
    embedding_dim = 64
).to(device)
data = data.to(device)

# Classifier Head
classifier = torch.nn.Linear(64, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(classifier.parameters()),
    lr = 0.01, weight_decay = 5e-4
)

print("\nStart Training...\n")
for epoch in range(200):
    model.train()
    classifier.train()
    optimizer.zero_grad()

    emb = model(data.x, data.edge_index)
    out = F.log_softmax(classifier(emb), dim=1)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    if epoch % 40 == 0 :
        model.eval()
        classifier.eval()
        with torch.no_grad():
            emb = model(data.x, data.edge_index)
            pred = classifier(emb).argmax(dim=1)
            acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
        print(f"Epoch : {epoch:03d} | Loss : {loss:.4f} | Test Acc : {acc:.4f}")
    
# Save Embedding
print("\nSaving Embedding...\n")
model.eval()
with torch.no_grad():
    embeddings = model(data.x, data.edge_index).cpu().numpy()

os.makedirs("data", exist_ok = True)

import numpy as np
np.save("data/embeddings.npy", embeddings)

papers = []
for i in range(data.num_nodes):
    papers.append({
        "id" : i,
        "title": paper_titles[i],
        "label": data.y[i].item(),
        "category": CLASS_NAMES[data.y[i].item()]
    })

with open("data/papers.json", "w", encoding= "utf-8") as f:
    json.dump(papers, f, ensure_ascii=False, indent=2)

edges = data.edge_index.t().cpu().numpy().tolist()
with open("data/edges.json", "w") as f:
    json.dump(edges[:500], f)

print("Save Completed!")
print("data/embeddings.npy -> GNN embedding")
print("data/papers.json -> paper meta data")
print("data/edges.json -> Citation relationship")
print("\nRun server by \'python RS_app.py\'")