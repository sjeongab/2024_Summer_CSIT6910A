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
            torch.nn.SELU(),
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
    
    def forward(self, fx, fy, W, H, pos, viewMatrix):
        focal_length_x = math.tan(fx * 0.5)
        focal_length_y = math.tan(fy * 0.5)
        W = int(W)
        H = int(H)
        i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
        dirs = torch.tensor(np.stack([(i-W*.5)/focal_length_x, -(j-H*.5)/focal_length_y, -np.ones_like(i)], -1), device="cuda")
        rays_d = torch.sum(dirs[..., np.newaxis, :] * viewMatrix, -1)
        rays_d = torch.nn.functional.normalize(rays_d)
        rays_o = pos.repeat((rays_d.shape[0], rays_d.shape[1], 1))
        directions = torch.cat((rays_o, rays_d), dim=-1)
        result = self.linear_stack(directions).permute([2,0,1])
        medium_rgb = torch.nn.Sigmoid(result[:3, :, :])
        backscatter = torch.nn.Softplus(result[3:,:,:])
        return medium_rgb, backscatter
    
    def get_output(self, camera):
        #input = [camera.FoVx, camera.FoVy, camera.image_width, camera.image_height, camera.camera_center, camera.world_view_transform[:3, :3]] 
        colour, backscatter = self.forward(camera.FoVx, camera.FoVy, camera.image_width, camera.image_height, camera.camera_center, camera.world_view_transform[:3, :3])
        return {"medium_rgb": colour, "medium_bs": backscatter}

    def save(self, path):
        mkdir_p(os.path.dirname(path))
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def export_to_onnx(self, path, camera):
        self.load_state_dict(torch.load(path))
        self.eval()
        input = (torch.tensor(camera.FoVx, device="cuda"), torch.tensor(camera.FoVy, device="cuda"), torch.tensor(camera.image_width, device="cuda"), torch.tensor(camera.image_height, device="cuda"), camera.camera_center, camera.world_view_transform[:3, :3]) 
        torch.onnx.export(self, input, "medium_model.onnx", verbose=True, input_names=["fx", "fy", "W", "H", "pos", "viewMatrix"], output_names=["medium_rgb", "backscatter"])

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
        return rays_d_encoded.type(torch.float)
    
    def forward(self, direction):
        return self.tcnn_encoding(direction).type(torch.float)
    
    def get_output(self, camera):
        direction = self.calculate_directions(camera)
        output = self.forward(direction).reshape([camera.image_height, camera.image_width, 6]).permute([2,0,1])
        colour = output[:3, ...]#self.out_activation(output[:3, ...])
        backscatter = output[3:6, ...]#self.out_activation(output[3:6, ...])
        return {"medium_rgb": colour, "medium_bs": backscatter}

    def save(self, path):
        mkdir_p(os.path.dirname(path))
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def export_to_onnx(self, path, camera=None):
        self.load_state_dict(torch.load(path))
        self.eval()
        if camera is None:
            direction = torch.zeros(1054*1600*6)
        else:
            direction = self.calculate_directions(camera)
        torch.onnx.export(self, direction, "medium_model.onnx", verbose=True, input_names=["directions"], output_names=["output"])