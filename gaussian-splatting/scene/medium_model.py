import torch
import math
import numpy as np
import os
from utils.system_utils import mkdir_p
import tinycudann as tcnn

class MediumModel(torch.nn.Module):
    def __init__(self):
        super(MediumModel, self).__init__()
        self.in_dim = 6
        self.out_dim = 6
        self.layer_width = 128
        self.activation = torch.nn.Softplus()
        self.out_activation = torch.nn.Sigmoid()

        self.linear_stack = torch.nn.Sequential(
            torch.nn.Linear(self.in_dim, self.layer_width),
            torch.nn.SoftPlus(),
            torch.nn.Linear(self.layer_width, self.out_dim),
        )
        
        colour = torch.tensor([])
        backscatter = torch.tensor([])
        self._colour = torch.nn.Parameter(colour.requires_grad_(True))
        self._backscatter = torch.nn.Parameter(backscatter.requires_grad_(True))

        l = [
            {'params': [self._colour], 'lr': 0.05, "name": "colour"},
            {'params': [self._backscatter], 'lr': 0.05, "name": "backscatter"},
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def calculate_directions(self, camera):
        focal_length_x = math.tan(camera.FoVx * 0.5)
        focal_length_y = math.tan(camera.FoVy * 0.5)
        W = camera.image_width
        H = camera.image_height
        i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
        dirs = torch.tensor(np.stack([(i-W*.5)/focal_length_x, -(j-H*.5)/focal_length_y, -np.ones_like(i)], -1), device="cuda")
        rays_d = torch.sum(dirs[..., np.newaxis, :] * camera.world_view_transform[:3, :3], -1)
        rays_d = torch.nn.functional.normalize(rays_d)
        rays_o = camera.camera_center.repeat((rays_d.shape[0], rays_d.shape[1], 1))
        return torch.cat((rays_o, rays_d), dim=-1)
    
    def forward(self, camera):
        x = self.calculate_directions(camera)        
        return self.linear_stack(x)
    
    def get_output(self, camera):
        output = self.forward(camera).permute([2,0,1])
        colour = output[:3, :, :]
        backscatter = output[3:, :, :]
        return {"medium_rgb": colour, "medium_bs": backscatter}

    def save(self, path):
        mkdir_p(os.path.dirname(path))
        torch.save(self.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path)
        new_state_dict = {}
        for k, v in state_dict.items():
            if "layers" not in k:
                new_state_dict[k] = v
        self.load_state_dict(new_state_dict)

class MediumTcnnModel(torch.nn.Module):
    def __init__(self):
        super(MediumTcnnModel, self).__init__()
        self.in_dim = 6
        self.out_dim = 6
        self.levels = 4
        self.layer_width = 128
        self.activation = torch.nn.Softplus()
        self.out_activation = torch.nn.Sigmoid()
        self.colour_bias = torch.tensor([0.0, 0.0, 0.2], device="cuda")

        colour = torch.tensor([], device="cuda")
        backscatter = torch.tensor([], device="cuda")
        self._colour = torch.nn.Parameter(colour.requires_grad_(True))
        self._backscatter = torch.nn.Parameter(backscatter.requires_grad_(True))

        l = [
            {'params': [self._colour], 'lr': 0.05, "name": "colour"},
            {'params': [self._backscatter], 'lr': 0.05, "name": "backscatter"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.loss = {"otype": "RelativeL2"}
        self.encoding = {"otype": "SphericalHarmonics",
                         "degree": self.levels}
        self.network = {
            #"otype": "FullyFusedMLP",
            "activation": "Tanh",
            "output_activation": "Sigmoid",
            "n_neurons": self.layer_width,
            "n_hidden_layers": 3
        }
        self.direction_encoding= tcnn.Encoding(
            n_input_dims=3,
            encoding_config=self.encoding
        )
        self.tcnn_encoding = tcnn.Network(
            n_input_dims=self.levels**2,
            n_output_dims=self.out_dim,
            network_config=self.network,
        )

    def calculate_directions(self, camera):
        focal_length_x = math.tan(camera.FoVx * 0.5)
        focal_length_y = math.tan(camera.FoVy * 0.5)
        W = camera.image_width
        H = camera.image_height
        i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
        dirs = torch.tensor(np.stack([(i-W*.5)/focal_length_x, -(j-H*.5)/focal_length_y, -np.ones_like(i)], -1), device="cuda")
        rays_d = torch.sum(dirs[..., np.newaxis, :] * camera.world_view_transform[:3, :3], -1)
        rays_d = (torch.nn.functional.normalize(rays_d)+1)/2.0
        rays_d_flat = rays_d.view(-1,3)
        rays_d_encoded = self.direction_encoding(rays_d_flat)
        return rays_d_encoded
    
    def forward(self, camera):
        direction = self.calculate_directions(camera)
        return self.tcnn_encoding(direction)
    
    def get_output(self, camera):
        output = self.forward(camera).reshape([camera.image_height, camera.image_width, 6]).permute([2,0,1])
        colour = output[:3, ...]#self.out_activation(output[:3, ...])
        backscatter = output[3:6, ...]#self.out_activation(output[3:6, ...])
        return {"medium_rgb": colour, "medium_bs": backscatter}

    def save(self, path):
        mkdir_p(os.path.dirname(path))
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))