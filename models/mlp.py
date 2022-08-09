import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, cfg):
        super(MLP, self).__init__()
        self.features = cfg.features
        self.body = nn.ModuleList()
        for i in range(len(self.features) - 1):
            self.body.append(nn.LayerNorm(self.features[i]))
            self.body.append(nn.Linear(self.features[i], self.features[i + 1]))
            self.body.append(nn.ReLU())

    def forward(self, x):
        out = x
        for module in self.body:
            out = module(out)
        return out
