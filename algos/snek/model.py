import torch.nn as nn
import torch.nn.functional as F


class Snek1DModel(nn.Module):

    def __init__(self, in_channels: int, kernel_size: int, out_features: int):
        super().__init__()
        self._in_channels = in_channels

        conv1 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels * 5, kernel_size=kernel_size, stride=max(kernel_size // 3, 1))
        conv2 = nn.Conv1d(in_channels=conv1.out_channels, out_channels=conv1.out_channels * 2, kernel_size=5, stride=2)

        conv3 = nn.Conv1d(in_channels=conv2.out_channels, out_channels=conv2.out_channels * 2, kernel_size=3)
        conv4 = nn.Conv1d(in_channels=conv3.out_channels, out_channels=conv3.out_channels, kernel_size=3)

        self._features = nn.Sequential(
            conv1, nn.ReLU(inplace=True), nn.BatchNorm1d(conv1.out_channels),
            conv2, nn.ReLU(inplace=True), nn.BatchNorm1d(conv2.out_channels),
            nn.MaxPool1d(3),
            conv3, nn.ReLU(inplace=True), nn.BatchNorm1d(conv3.out_channels),
            conv4, nn.ReLU(inplace=True), nn.BatchNorm1d(conv4.out_channels),
            nn.AvgPool1d(conv4.out_channels),
            nn.Dropout(0.2)
        )
        self.head = nn.Linear(conv4.out_channels, out_features=out_features)

    def forward(self, tensor):
        x = tensor.view(1, self._in_channels, -1).float()
        x = self._features(x)
        x = self.head(x.view(x.size(0), -1))
        return F.softmax(x, dim=0)
