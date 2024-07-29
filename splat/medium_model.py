import torch
import math
import numpy as np
import os
from utils.system_utils import mkdir_p

class MediumModel(torch.nn.Module):
    def __init__(self):
        super(MediumModel, self).__init__()
        self.in_dim = 6
        self.out_dim = 6
        self.layer_width = 128
        self.activation = torch.nn.Softplus()
        self.out_activation = torch.nn.Sigmoid()

        layers = []
        layers.append(torch.nn.Linear(self.in_dim, self.layer_width))
        #layers.append(torch.nn.Linear(self.layer_width, self.layer_width))
        layers.append(torch.nn.Linear(self.layer_width, self.out_dim))
        for name, param in self.named_parameters():
            if param.requires_grad:
                print (name, param.data)

        #self.colour = torch.tensor([])
        #self.backscatter = torch.tensor([])
        self.layers = torch.nn.ModuleList(layers)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0, eps=1e-15)

    #TODO: revisit calculate_directions
    def calculate_directions(self, camera):
        focal_length_x = math.tan(camera.FoVx * 0.5)
        focal_length_y = math.tan(camera.FoVy * 0.5)
        W = camera.image_width
        H = camera.image_height
        i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
        dirs = torch.tensor(np.stack([(i-W*.5)/focal_length_x, -(j-H*.5)/focal_length_y, -np.ones_like(i)], -1), device="cuda")
        rays_d = torch.sum(dirs[..., np.newaxis, :] * camera.world_view_transform[:3, :3], -1)

        #directions = get_normalized_directions(torch.nn.functional.normalize(rays_d))
        #directions_flat = directions.view(-1, 3)
        #directions_encoded = self.direction_encoding(directions_flat)
        #return directions_encoded
        rays_o = camera.camera_center.repeat((rays_d.shape[0], rays_d.shape[1], 1))
        return torch.cat((rays_o, rays_d), dim=-1)
    
    def forward(self, camera):
        x = self.calculate_directions(camera)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.activation is not None and i < len(self.layers) - 1:
                x = self.activation(x)
        if self.out_activation is not None:
            x = self.out_activation(x)
        return x
    
    def get_output(self, camera):
        output = self.forward(camera)
        colour = output[:, :, :3]#.mean(dim=0).mean(dim=0)#.permute([2,0,1])
        backscatter = output[:, :, 3:]#.permute([2,0,1])
        return {"medium_rgb": colour, "medium_bs": backscatter}

    def save(self, path):
        mkdir_p(os.path.dirname(path))
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))