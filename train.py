import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from models.gcn import GCN

dataset = Planetoid(root='data', name='Cora')
data = dataset[0]

print(f"Nodes: {data.num_nodes}")
print(f"Edges: {data.num_edges}")
print(f"Classes: {dataset.num_classes}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(
    in_channels=dataset.num_node_features,
    hidden_channels=64,
    out_channels=dataset.num_classes
).to(device)
data = data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay = 5e-4)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate():
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)
        acc = (pred[data.test_mask] ==data.y[data.test_mask]).float().mean()
    return acc.item()

for epoch in range(200):
    loss = train()
    if epoch % 20 ==0:
        acc = evaluate()
        print(f"Epoch {epoch:03d} | Loss : {loss:.4f} | Test Acc: {acc:.4f}")