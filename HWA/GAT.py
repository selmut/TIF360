from torch_geometric.nn import GCNConv, GATConv, Linear, TopKPooling, SAGPooling, SAGEConv
import torch.nn.functional as F
import torch


class GNNNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNNNet, self).__init__()
        #torch.manual_seed(12345)
        self.gat_conv1 = GATConv(in_channels, 64)
        self.gat_conv2 = GATConv(64, 128)
        # self.sage_conv1 = SAGEConv(128, 128)
        self.gat_conv3 = GATConv(128, 128)
        # self.topk_pool = TopKPooling(128, ratio=1)
        self.sag_pool = SAGPooling(128, ratio=1)
        self.lin = Linear(128, out_channels)

    def forward(self, x, edge_index, batch, edge_attr):
        # 1. Obtain node embeddings
        x = self.gat_conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = self.gat_conv2(x, edge_index, edge_attr)
        x = x.relu()
        # x = self.sage_conv1(x, edge_index)
        x = self.gat_conv3(x, edge_index, edge_attr)
        x = x.relu()

        # 2. Readout layer
        #(x, edge_index, edge_attr, batch, perm, _) = self.topk_pool(x, edge_index, edge_attr, batch)
        (x, edge_index, edge_attr, batch, perm, _) = self.sag_pool(x, edge_index, edge_attr, batch)

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        x = x.relu()
        return x


class GNN:
    def __init__(self, in_channels, out_channels):
        self.model = GNNNet(in_channels=in_channels, out_channels=out_channels)
        self.criterion = torch.nn.MSELoss()  # Define loss criterion.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0008, amsgrad=True)  # Define optimizer.
        self.out_channels = out_channels

    def fn_train(self, train_loader):
        self.model.train()

        for data in train_loader:  # Iterate in batches over the training dataset.
            out = self.model(data.x, data.edge_index, data.batch, data.edge_attr)  # Perform a single forward pass.
            # loss = self.criterion(torch.round(out), data.y)  # Compute the loss.
            loss = self.criterion(out, data.y[:, 0:1])  # Compute the loss.
            loss.backward()  # Derive gradients.
            self.optimizer.step()  # Update parameters based on gradients.
            self.optimizer.zero_grad()  # Clear gradients.

    def fn_test(self, loader):
        self.model.eval()

        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            out = self.model(data.x, data.edge_index, data.batch, data.edge_attr)
            pred = torch.round(out)
            correct = (pred == data.y[:, 0:1]).all(dim=1)
        return int(correct.sum())/int(pred.size()[0])

    def fn_load_model(self, model_name):
        self.model = torch.load(model_name)
