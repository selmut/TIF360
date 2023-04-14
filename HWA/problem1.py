from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from plots import *
from MLP import MLP
from GCN import GCN
import torch

dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
data = dataset[0]  # Get the first graph object.

'''print()
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

print()
print(data)
print('===========================================================================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}\n')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Number of validation nodes: {data.val_mask.sum()}')
print(f'Number of test nodes: {data.test_mask.sum()}\n')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')'''


mlp = MLP(16, dataset)

# train mlp
for epoch in range(1, 201):
    loss = mlp.fn_train(data)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

gcn = GCN(16, dataset)

for epoch in range(1, 101):
    loss = gcn.fn_train(data)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

mlp_out = mlp.model(data.x)
visualize(mlp_out, data.y, f'img/mlp.png')

gcn_out = gcn.model(data.x, data.edge_index)
visualize(gcn_out, data.y, f'img/gcn.png')
