import torch
import torch.nn as nn

class DeepSVDD(nn.Module):
    def __init__(self, feature_extractor, latent_dim):
        super(DeepSVDD, self).__init__()
        self.feature_extractor = feature_extractor
        self.center = torch.zeros(latent_dim).to("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        features = self.feature_extractor(x)
        return features

    def loss(self, features):
        return torch.mean(torch.sum((features - self.center) ** 2, dim=1))
