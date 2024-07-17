import torch
from typing import Dict, Literal
from nerfstudio.field_components.encodings import HashEncoding, SHEncoding
from nerfstudio.field_components.mlp import MLP

class MediumModel:
    def __init__(self,
                 implementation: Literal["tcnn", "torch"] = "tcnn",
                 num_layers_medium: int = 2,
                 hidden_dim_medium: int = 128,
                 medium_density_bias: float = 0.0,
                 ):
        self.medium_colour = torch.tensor([1/255, 50/255, 32/255], device="cuda")
        self.medium_density_bias = medium_density_bias
        self.direction_encoding = SHEncoding(levels=4, implementation=implementation)
        self.colour_activation = torch.nn.Sigmoid()
        self.sigma_activation = torch.nn.Softplus()

        self.medium_mlp = MLP(
            in_dim=self.direction_encoding.get_out_dim(),
            num_layers=num_layers_medium,
            layer_width=hidden_dim_medium,
            out_dim=9,
            activation=torch.nn.Softplus(),
            out_activation=None,
            implementation=implementation,
        )