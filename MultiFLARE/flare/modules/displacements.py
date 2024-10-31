# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import torch

class Displacements(torch.nn.Module):
    def __init__(self, vertices_shape=(5023, 3)):
        super().__init__()
        self.vertices_shape = vertices_shape
        self.register_parameter("vertex_offsets", torch.nn.Parameter(torch.zeros(vertices_shape), requires_grad=True))

    def forward(self):
        return self.vertex_offsets

    def save(self, path):
        # Store the config
        _config = {
            "vertices_shape":self.vertices_shape,
        }

        data = {
            'version': 2,
            'config': _config,
            'state_dict': self.state_dict()
        }
        torch.save(data, path)

    @classmethod
    def load(cls, path, device='cpu'):
        data = torch.load(path, map_location=device)

        displacements = cls(**data['config'], device=device)
        displacements.load_state_dict(data['state_dict'])

        return displacements
