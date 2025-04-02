#需要修改！！！现在固定的输出为单INR双输出

import math
import torch
import numpy as np
import torch.nn as nn


class ReLUActivationLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=20, sigma=1):
        super(ReLUActivationLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0.0)

    # FMLP对ReLU层的初始化 适用于FourierFeature sigma是一个可变频率 作为超参数调制
    # def init_weights(self):
    #    with torch.no_gard():
    #        self.linear.weight.normal_(std=self.sigma)
    #        if self.linear.bias is not None:
    #            self.linear.bias.normal_(std=1e-6)

    def forward(self, x):
        return torch.relu(self.linear(x))


class SineActivationLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30):
        super(SineActivationLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.omega_0 = omega_0
        self.is_first = is_first
        self.init_weights(in_features)

    def init_weights(self, in_features):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / in_features), np.sqrt(6 / in_features))

    def forward(self, x):
        if self.is_first:
            return torch.sin(self.omega_0 * self.linear(x))
        else:
            return torch.sin(self.linear(x))

def get_activation_layer(in_features, out_features, activation, is_first, omega_0):
    if activation.lower() == "relu":
        return ReLUActivationLayer(in_features, out_features, is_first=is_first, omega_0=omega_0)
    elif activation.lower() == "sine":
        return SineActivationLayer(in_features, out_features, is_first=is_first, omega_0=omega_0)
    else:
        raise ValueError("Unsupported activation: " + activation)

class FourierFeatureMap(nn.Module):
    def __init__(self, in_features, out_features, coordinate_scales):
        super(FourierFeatureMap, self).__init__()
        assert out_features % 2 == 0, "Fourier Features not even number!"
        self.num_freq = out_features // 2
        self.out_features = out_features
        self.coordinate_scales = nn.Parameter(torch.tensor(coordinate_scales, dtype=torch.float32).unsqueeze(0),
                                              requires_grad=False)
        self.B = torch.normal(0, 3, (in_features, self.num_freq)) * 10  # (mean, std, size)

        self.B = nn.Parameter(self.B, requires_grad=False)

    def forward(self, x):
        scaled = self.coordinate_scales * x
        proj = scaled @ self.B
        sin_feat = np.sqrt(2) * torch.sin(proj)
        cos_feat = np.sqrt(2) * torch.cos(proj)
        return torch.cat((sin_feat, cos_feat), dim=-1)

class ExpertMLP(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, omega_0, activation):
        super(ExpertMLP, self).__init__()
        layers = []
        first_layer = get_activation_layer(in_features, hidden_features, activation, is_first=True, omega_0=omega_0)
        layers.append(first_layer)
        for i in range(hidden_layers):
            layer = get_activation_layer(hidden_features, hidden_features, activation, is_first=False, omega_0=omega_0)
            layers.append(layer)
        final_layer = nn.Linear(hidden_features, out_features)
        layers.append(final_layer)
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class Fullmodel(nn.Module):
    def __init__(self,
                 encoding_mode,
                 in_features, out_features, coordinate_scales,
                 mlp_hidden_features, mlp_hidden_layers,
                 omega_0, activation):# , inr_predict_separate_together, use_moe, num_experts
        super(Fullmodel, self).__init__()
        self.encoding_mode = encoding_mode.lower()
        if self.encoding_mode == "fourier":
            self.encoder = FourierFeatureMap(
                in_features, out_features, coordinate_scales)
            encoder_output_dim = out_features
        else:
            raise ValueError("Unsupported encoding_mode: " + encoding_mode)
        self.net = ExpertMLP(encoder_output_dim, mlp_hidden_features, mlp_hidden_layers, 2, omega_0, activation)

    def forward(self, x):
        encoded = self.encoder(x)
        out = self.net(encoded)
        return out

