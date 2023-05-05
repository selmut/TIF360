from torch_geometric.nn import GCNConv, GATConv, Linear, SAGPooling, TopKPooling
import torch.nn.functional as F
import torch
import numpy as np


class GNNNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNNNet, self).__init__()
        self.n_heads = 4

        self.gat_conv1 = GATConv(in_channels, 64, heads=self.n_heads)
        self.gat_conv2 = GATConv(64*self.n_heads, 64, heads=self.n_heads)
        self.gat_conv3 = GATConv(64*self.n_heads, 64, heads=self.n_heads)
        self.sag_pool = SAGPooling(64*self.n_heads, ratio=1)
        self.topk_pool = TopKPooling(64*self.n_heads, ratio=1)
        self.lin = Linear(64*self.n_heads, out_channels)

    def forward(self, x, edge_index, batch, edge_attr):
        x = self.gat_conv1(x, edge_index, edge_attr)
        x = x.relu()
        add = x
        x = self.gat_conv2(x, edge_index, edge_attr)
        x = x.relu()
        add = add+x
        x = self.gat_conv3(x, edge_index, edge_attr)
        x = x.relu()

        add = add + x

        (x, edge_index, edge_attr, batch, perm, _) = self.sag_pool(add, edge_index, edge_attr, batch)
        # (x, edge_index, edge_attr, batch, perm, _) = self.topk_pool(add, edge_index, edge_attr, batch)

        x = F.dropout(x, p=0.4, training=self.training)
        x = self.lin(x)
        x = x.sigmoid()
        return x


class GNN:
    def __init__(self, in_channels, out_channels):
        self.model = GNNNet(in_channels=in_channels, out_channels=out_channels)
        self.criterion = torch.nn.BCELoss()  # Define loss criterion.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.002, amsgrad=True)  # Define optimizer.
        self.out_channels = out_channels

    def fn_train(self, train_loader):
        self.model.train()

        for data in train_loader:  # Iterate in batches over the training dataset.
            out = self.model(data.x, data.edge_index, data.batch, data.edge_attr)  # Perform a single forward pass.
            # loss = self.criterion(torch.round(out), data.y)  # Compute the loss.
            loss = self.criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            self.optimizer.step()  # Update parameters based on gradients.
            self.optimizer.zero_grad()  # Clear gradients.

    def fn_test(self, loader):
        self.model.eval()
        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            out = self.model(data.x, data.edge_index, data.batch, data.edge_attr)
            pred = torch.round(out).detach().numpy()
            labels = data.y.numpy()

            diff = np.sum(np.abs(pred-labels), axis=1)
            correct += len(np.where(diff == 0)[0])/len(data.y)
        return correct/len(loader)

    def fn_load_model(self, model_name):
        self.model = torch.load(model_name)
