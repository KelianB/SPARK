## Code: https://github.com/zhengyuf/IMavatar
# Modified/Adapted by: Shrisha Bharadwaj

import torch
from flare.modules.embedder import *
import numpy as np
import torch.nn as nn
import os

class ForwardDeformer(nn.Module):
    def __init__(self,
                flame,
                dims,
                multires,
                num_exp=50,
                aabb=None,
                weight_norm=True,
                deformer_input="canonical_pos",
                overrides=False,
                expr_only=False,
            ):
        super().__init__()

        self._config = {
            "dims":dims,
            "multires":multires,
            "num_exp":num_exp,
            "aabb":aabb,
            "weight_norm":weight_norm,
            "deformer_input":deformer_input,
            "overrides": overrides,
            "expr_only": expr_only,
        }

        self.flame = flame
        #  ============================== pose correctives, expression blendshapes and linear blend skinning weights  ==============================
        d_in = 3 if deformer_input == "canonical_pos" else 2
        d_out = 36 * 3 + num_exp * 3 

        self.num_exp = num_exp
        dims = [d_in] + dims + [d_out]
        self.embed_fn = None
        if multires > 0:
            self.embed_fn, input_ch = get_embedder(multires)
            dims[0] = input_ch
            
        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 2):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if multires > 0 and l == 0:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
            else:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)
        self.blendshapes = nn.Linear(dims[self.num_layers - 2], d_out)
        self.skinning_linear = nn.Linear(dims[self.num_layers - 2], dims[self.num_layers - 2])
        self.skinning = nn.Linear(dims[self.num_layers - 2], flame.n_joints)
        torch.nn.init.constant_(self.skinning_linear.bias, 0.0)
        torch.nn.init.normal_(self.skinning_linear.weight, 0.0, np.sqrt(2) / np.sqrt(dims[self.num_layers - 2]))
        if weight_norm:
            self.skinning_linear = nn.utils.weight_norm(self.skinning_linear)
        ## ==============  initialize blendshapes to be zero, and skinning weights to be equal for every bone (after softmax activation)  ==============================
        torch.nn.init.constant_(self.blendshapes.bias, 0.0)
        torch.nn.init.constant_(self.blendshapes.weight, 0.0)
        torch.nn.init.constant_(self.skinning.bias, 0.0)
        torch.nn.init.constant_(self.skinning.weight, 0.0)

        self.aabb = aabb


    def query_weights(self, pnts_c):
        # Normalize input
        pnts_c = (pnts_c.view(-1, 3) - self.aabb[0].unsqueeze(0)) / (self.aabb[1].unsqueeze(0) - self.aabb[0].unsqueeze(0))

        # Embedding
        if self.embed_fn is not None:
            # pnts_c = torch.clamp(pnts_c, min=0, max=1)
            pnts_c = self.embed_fn(pnts_c).float()

        x = pnts_c

        for l in range(0, self.num_layers - 2):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            x = self.softplus(x)

        P = (self.flame.n_joints - 1) * 9 # number of vectors per posedir

        blendshapes = self.blendshapes(x)
        posedirs = blendshapes[:, :P * 3]
        shapedirs = blendshapes[:, P * 3: P * 3 + self.num_exp * 3]
        lbs_weights = self.skinning(self.softplus(self.skinning_linear(x)))
        lbs_weights_exp = torch.exp(20 * lbs_weights)
        lbs_weights = lbs_weights_exp / torch.sum(lbs_weights_exp, dim=-1, keepdim=True)

        # Format the tensors as expected by the FLAME model
        shapedirs = shapedirs.view(-1, 3, self.num_exp)
        lbs_weights = lbs_weights.view(-1, self.flame.n_joints)
        posedirs = posedirs.view(-1, P, 3) # (V, P*3) to (V, P, 3)

        # Manually set parts whose values we don't want to learn.
        # This is optional, but it does tend to resolve artifacts on the geometry in some cases.
        if self.overrides:
            # Force values for the teeth
            flame = self.flame
            if flame.has_teeth:
                # Teeth have no shapedirs or posedirs
                shapedirs[flame.mask.v.teeth] = 0
                posedirs[flame.mask.v.teeth] = 0
                lbs_weights[flame.mask.v.teeth] = 0
                # Upper teeth are only linked to the neck joint
                lbs_weights[flame.mask.v.teeth_upper, 1] = 1
                # Lower teeth are only linked to the jaw joint
                lbs_weights[flame.mask.v.teeth_lower, 2] = 1

            # Force values for the eyes
            eye_verts_joint = [(flame.mask.v.left_eyeball, 3), (flame.mask.v.right_eyeball, 4)]
            for (eye_verts, joint) in eye_verts_joint:
                # Eye joints have no effect outside the eyeballs
                lbs_weights[:, joint] = 0
                posedirs.view(-1, self.flame.n_joints-1, 9, 3)[:, joint-1] = 0
                
                # Eyes only have LBS weights, no shapedirs or posedirs
                shapedirs[eye_verts] = 0 
                posedirs[eye_verts] = 0
                lbs_weights[eye_verts] = 0
                lbs_weights[eye_verts, joint] = 1        

            # Override shapedirs for the eyelids (helps avoid artifacts in some cases)
            left_eyelid_mask = flame.mask.f.left_eyelid_extended[:, None, None]
            right_eyelid_mask = flame.mask.f.right_eyelid_extended[:, None, None]
            shapedirs = flame.shapedirs_expression_updated * left_eyelid_mask + shapedirs * (1 - left_eyelid_mask)
            shapedirs = flame.shapedirs_expression_updated * right_eyelid_mask + shapedirs * (1 - right_eyelid_mask)

        posedirs = posedirs.permute(1, 0, 2).reshape(P, -1) # (V, P, 3) to (P, 3*V)

        if self.expr_only:
            posedirs = self.flame.posedirs_updated
            lbs_weights = self.flame.lbs_weights_updated
            
        return shapedirs, posedirs, lbs_weights

    @property
    def input(self):
        return self._config["deformer_input"]

    @property
    def overrides(self):
        return self._config["overrides"]

    @overrides.setter
    def overrides(self, b: bool):
        self._config["overrides"] = b
    
    @property
    def expr_only(self):
        return self._config["expr_only"]

    @expr_only.setter
    def expr_only(self, b: bool):
        self._config["expr_only"] = b

    def clone(self, device):
        new_deformer = ForwardDeformer(self.flame, **self._config).to(device)
        new_deformer.load_state_dict(self.state_dict(), strict=False)
        return new_deformer

    @classmethod
    def revive(cls, path, flame, device='cpu'):
        assert os.path.exists(path)
        data = torch.load(path, map_location=device)
        deformer = cls(flame, **data['config']).to(device)
        deformer.load_state_dict(data['state_dict'], strict=False)
        return deformer

    def load(self, path, device='cpu'):
        assert os.path.exists(path)
        data = torch.load(path, map_location=device)
        self.load_state_dict(data['state_dict'], strict=False)

    def save(self, path):
        # remove flame temporarily so we don't save it
        self.flame, flame = None, self.flame 
        data = {
            'config': self._config,
            'state_dict': self.state_dict()
        }
        self.flame = flame
        torch.save(data, path)
