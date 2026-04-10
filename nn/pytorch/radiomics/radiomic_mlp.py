import numpy as np
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class StatisticalRadiomicTransformer(torch.nn.Module):
    def __init__(self, device, n_classes: int = 2, n_head=8):
        super(StatisticalRadiomicTransformer, self).__init__()
        self.mins = None
        self.maxs = None
        self.device = device
        encoder_layer = TransformerEncoderLayer(d_model=features_layer.output_size, nhead=n_head)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=3)
        self.feed_forward = torch.nn.Linear(self.features_layer.output_size, n_classes)

    def init(self, dataset) -> None:
        features = torch.stack([self.features_layer.forward(torch.FloatTensor(x).to(self.device)).reshape(self.features_layer.output_size) for x in dataset], dim=0)
        self.mins = torch.min(features, dim=0)
        self.mins.values.requires_grad = False
        self.maxs = torch.max(features, dim=0)
        self.maxs.values.requires_grad = False

    def forward(self, x):
        features = torch.nan_to_num(self.features_layer(x))
        features = (features - self.mins.values) / (self.maxs.values - self.mins.values)
        t = self.transformer_encoder(features)
        return self.feed_forward(t)

class StatisticalRadiomicMLP(torch.nn.Module):
    def __init__(self, device, in_features: int, n_classes: int = 2) -> None:
        super().__init__()
        self.mins = None
        self.maxs = None
        self.device = device
        self.hidden_size = 128
        self.input_size = in_features
        self.classification_model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, n_classes)
        )

    """
    Pre inits MLP by calculating min and max values of resulting statistical radiomic features 
    to apply normalization later
    """
    def init(self, dataset) -> None:
        pass

    def forward(self, x):
        return self.classification_model(x)