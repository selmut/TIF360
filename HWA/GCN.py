from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch


class GCNNet(torch.nn.Module):
    def __init__(self, hidden_channels, dataset):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GCN:
    def __init__(self, hidden_channels, dataset):
        self.model = GCNNet(hidden_channels=hidden_channels, dataset=dataset)
        self.criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)  # Define optimizer.

    def fn_train(self, data):
        self.model.train()
        self.optimizer.zero_grad()  # Clear gradients.
        out = self.model(data.x, data.edge_index)  # Perform a single forward pass.
        loss = self.criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        self.optimizer.step()  # Update parameters based on gradients.
        return loss

    def fn_test(self, data):
        self.model.eval()
        out = self.model(data.x, data.edge_index)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
        test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
        return test_acc
