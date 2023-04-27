import torch
from torch.nn import Linear
import torch.nn.functional as F


class MLPNet(torch.nn.Module):
    def __init__(self, hidden_channels, dataset):
        super().__init__()
        #torch.manual_seed(12345)
        self.lin1 = Linear(dataset.num_features, hidden_channels)
        self.lin2 = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x


class MLP:
    def __init__(self, hidden_channels, dataset):
        self.model = MLPNet(hidden_channels=hidden_channels, dataset=dataset)
        self.criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)  # Define optimizer.

    def fn_train(self, data):
        self.model.train()
        self.optimizer.zero_grad()  # Clear gradients.
        out = self.model(data.x)  # Perform a single forward pass.
        loss = self.criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        self.optimizer.step()  # Update parameters based on gradients.
        return loss

    def fn_test(self, data):
        self.model.eval()
        out = self.model(data.x)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
        test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
        return test_acc

