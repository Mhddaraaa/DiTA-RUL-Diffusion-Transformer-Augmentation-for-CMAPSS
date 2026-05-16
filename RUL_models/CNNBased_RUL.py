import torch
import torch.nn as nn


class CNNRULPredictor(nn.Module):
    def __init__(self, in_c=1, ws=30, in_features=17):
        super(CNNRULPredictor, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_c, out_channels=32, kernel_size=(in_features, 3), stride=(2, 2))
        self.gelu = nn.GELU()

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), stride=(2, 2))

        # Calculate feature map size dynamically
        feature_map_size = self.get_feature_map_size((in_c, ws, in_features))

        self.fc1 = nn.Linear(feature_map_size, 64)  # feature_map_size depends on input dimensions
        self.dropout1 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.1)

        self.fc3 = nn.Linear(32, 1)

    def get_feature_map_size(self, input_shape):
        x = torch.zeros(1, *input_shape)
        x = self.conv1(x)
        x = self.conv2(x)
        return x.shape[1] * x.shape[2] * x.shape[3]

    def forward(self, x):
        x = self.conv1(x.unsqueeze(1))
        x = self.gelu(x)

        x = self.conv2(x)
        x = self.gelu(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.gelu(x)
        x = self.dropout2(x)

        x = self.fc3(x)

        return x
