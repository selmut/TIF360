import numpy as np
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.loader import DataLoader
from plots import *
from MLP import MLP
from GAT import GNN
import torch
import pandas as pd

print('Loading data...')
dataset = torch.load('data/graph_data_decoding_2023.pt')
# data = dataset[0]  # Get the first graph object.

print('Generating test/train splits...')
num_samples = len(dataset)
train_data = dataset[int(.05 * num_samples):]  # 95% train
test_data = dataset[: int(.05 * num_samples)]  # 5% test

train_loader = DataLoader(train_data, batch_size=1000, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000, shuffle=True)


print('====================')
print(f'Number of graphs: {len(dataset)}')
print('=============================================================')
data = dataset[20]  # Get one data point.

print('Some properties of a graph in the dataset:')
print()

print(data)
# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')
print(f'Number of node features: {data.num_node_features}')

label = data.y
label_X = label[:, 0]
label_Z = label[:, 1]

gnn = GNN(4, 2)
epochs = 10

train_accs = np.zeros(epochs)
test_accs = np.zeros(epochs)

# train gcn
print('\nTraining GCN...')
for epoch in range(0, epochs):
    gnn.fn_train(train_loader)
    train_acc = gnn.fn_test(train_loader)
    test_acc = gnn.fn_test(test_loader)

    train_accs[epoch] = train_acc
    test_accs[epoch] = test_acc

    if test_acc >= 0.9:
        torch.save(gnn.model, f'models/test_acc{test_acc:.4f}_train_acc{train_acc:.4f}.pt')

    print(f'Epoch: {epoch+1:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')


pd.DataFrame(train_accs).to_csv('csv/train_accuracies.csv')
pd.DataFrame(test_accs).to_csv('csv/test_accuracies.csv')


