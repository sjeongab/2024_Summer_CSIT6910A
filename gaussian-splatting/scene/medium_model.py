import torch
import os
from typing import Dict, Literal
from utils.system_utils import mkdir_p
from nerfstudio.field_components.encodings import HashEncoding, SHEncoding
from nerfstudio.field_components.mlp import MLP

class MediumModel(torch.nn.Module):
    def __init__(self,
                 implementation: Literal["tcnn", "torch"] = "tcnn",
                 num_layers_medium: int = 2,
                 hidden_dim_medium: int = 128,
                 medium_density_bias: float = 0.0,
                 ):
        super().__init__()

        self.medium_colour = torch.tensor([1/255, 50/255, 32/255], device="cuda")
        self.medium_density_bias = medium_density_bias
        #self.direction_encoding = SHEncoding(levels=4, implementation=implementation)
        self.colour_activation = torch.nn.Sigmoid()
        self.sigma_activation = torch.nn.Softplus()
        #print(self.direction_encoding.get_out_dim())
        #print("medium_model 21")
        self.medium_mlp = MLP(
            in_dim = 3, #self.direction_encoding.get_out_dim(),
            num_layers=num_layers_medium,
            layer_width=hidden_dim_medium,
            out_dim=4,
            activation=torch.nn.Softplus(),
            out_activation=None,
            implementation=implementation,
        )

    def get_outputs(self, directions):
        #TODO: add cam_dir encoding?

        outputs = {}
        medium_base_out = self.medium_mlp(directions)

        medium_rgb = (
            self.colour_activation(medium_base_out[..., :3])
            #.view(*outputs_shape, -1)
            .to(directions)
        )
        medium_bs = (
            self.sigma_activation(medium_base_out[..., 3] + self.medium_density_bias)
            #.view(*outputs_shape, -1)
            .to(directions)
        )
        """medium_attn = (
            self.sigma_activation(medium_base_out[..., 6:] + self.medium_density_bias)
            .view(*outputs_shape, -1)
            .to(directions)
        )"""

        outputs["medium_colour"] = medium_rgb
        outputs["medium_bs"] = medium_bs
        #outputs[SeathruHeadNames.MEDIUM_ATTN] = medium_attn

        return outputs
    
    def forward(self, directions):
        """base_field"""
        """if compute_normals:
            with torch.enable_grad():
                density, density_embedding = self.get_density(ray_samples)
        else:
            density, density_embedding = self.get_density(ray_samples)"""

        field_outputs = self.get_outputs(directions)

        return field_outputs
    
    def save(self, path):
        mkdir_p(os.path.dirname(path))
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
