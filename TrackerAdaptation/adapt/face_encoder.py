from typing import Dict

import torch

from adapt.deca_helpers import create_deca
from adapt.common_types import EncodedValues
from adapt.constants import SMIRK_PATH

class FaceEncoder(torch.nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

    def set_train_mode(self, b: bool):
        raise NotImplementedError()

    def forward(self, views_crop: Dict, training: bool) -> EncodedValues:
        raise NotImplementedError()


class DECAEncoder(FaceEncoder):
    def __init__(self, device: torch.device, checkpoints_dir: str, deca_model: str, deca_cfg: str, resume: int,
                 deca_detail: bool, inference_only=False):
        super().__init__(device)
        self.checkpoints_dir = checkpoints_dir
        self.deca_detail = deca_detail

        # Create EMOCA model
        checkpoint = (checkpoints_dir / f"deca_iter{resume:04d}.pth") if resume > 0 else None
        deca = create_deca(deca_model, deca_cfg, deca_detail, inference_only=inference_only, checkpoint=checkpoint)
        self.deca = deca.to(device)

        self.faces = deca.deca.render.faces.squeeze(0) #deca.deca.flame.faces_tensor
        self.n_verts = deca.deca.flame.v_template.shape[0]

    def set_train_mode(self, b: bool):
        self.deca.deca.train(b)
    
    def save_checkpoint(self, iteration: int):
        from pytorch_lightning.utilities.cloud_io import atomic_save
        checkpoint = {"state_dict": self.deca.state_dict()}
        checkpoint_path = self.checkpoints_dir / f"deca_iter{iteration:04d}.pth"
        atomic_save(checkpoint, checkpoint_path)

    def get_trainables(self, args):
        deca = self.deca.deca
        t = []
        if args.train_backbones:
            t += [deca.E_flame.encoder, deca.E_expression.encoder]
        if args.train_backbones_last:
            t += [deca.E_flame.encoder.layer4, deca.E_expression.encoder.layer4]
        if args.train_mlps:
            t += [deca.E_flame.layers, deca.E_expression.layers]
        return t

    def forward(self, views_crop: Dict, training: bool) -> EncodedValues:
        views_deca = {**views_crop}
        if training:
            # For DECA's shape consistency loss, we need images to have shape (1,K,...) where K is the ring size (number of images of the same subject)
            # In our case, all images are of the same subject.
            views_deca["image"] = views_deca["image"].unsqueeze(0)
        deca_v = self.deca.encode(views_deca, training=training)
        if not training:
            # Since we're predicting the shape of the same person, we average the shape code over the batch
            deca_v["shapecode"] = deca_v["shapecode"].mean(0, keepdim=True).expand_as(deca_v["shapecode"])

        values = EncodedValues()
        values.expcode = deca_v["expcode"]
        values.posecode = deca_v["posecode"]
        values.cam = deca_v["cam"]
        values.shapecode = deca_v["shapecode"]
        values.texcode = deca_v["texcode"]
        values.lightcode = deca_v["lightcode"]
        if "detailcode" in deca_v:
            values.detailcode = deca_v["detailcode"]
            values.detailemocode = deca_v["detailemocode"]
        values.eyelids = None
        values.images = deca_v["images"]
        values.lmk = deca_v["lmk"]
        values.lmk_mp = deca_v["lmk_mp"]
        values.masks = deca_v["masks"]
        return values


class SMIRKEncoder(FaceEncoder):
    def __init__(self, device, checkpoints_dir: str, resume: int, inference_only=False):
        super().__init__(device)
        self.checkpoints_dir = checkpoints_dir

        if resume > 0:
            checkpoint = checkpoints_dir / f"smirk_iter{resume:04d}.pth"
            checkpoint = torch.load(checkpoint)
            checkpoint_encoder = checkpoint["encoder"]
            checkpoint_generator = checkpoint["generator"]
        else:
            checkpoint = str(SMIRK_PATH / "pretrained_models/SMIRK_em1.pt")
            checkpoint = torch.load(checkpoint)
            checkpoint_encoder = {k.replace('smirk_encoder.', ''): v for k, v in checkpoint.items() if 'smirk_encoder' in k} # checkpoint includes both smirk_encoder and smirk_generator
            checkpoint_generator = {k.replace('smirk_generator.', ''): v for k, v in checkpoint.items() if 'smirk_generator' in k} # checkpoint includes both smirk_encoder and smirk_generator
        
        # SMIRK imports
        from src.smirk_encoder import SmirkEncoder
        from src.smirk_generator import SmirkGenerator

        self.smirk_encoder = SmirkEncoder().to(device)
        self.smirk_encoder.load_state_dict(checkpoint_encoder)
        self.smirk_generator = SmirkGenerator(in_channels=6, out_channels=3, init_features=32, res_blocks=5).to(self.device)
        self.smirk_generator.load_state_dict(checkpoint_generator)


    def set_train_mode(self, b: bool):
        self.smirk_encoder.train(b)
        self.smirk_generator.train(b)

    def get_trainables(self, args):
        enc = self.smirk_encoder
        t = []
        if args.train_backbones:
            t += [enc.pose_encoder.encoder, enc.expression_encoder.encoder, enc.shape_encoder.encoder]
        if args.train_backbones_last:
            t += [enc.pose_encoder.encoder.blocks[-1], enc.expression_encoder.encoder.blocks[-1], enc.shape_encoder.encoder.blocks[-1]]
        if args.train_mlps:
            t += [enc.pose_encoder.pose_cam_layers, enc.expression_encoder.expression_layers, enc.shape_encoder.shape_layers]
        t += [self.smirk_generator]
        return t

    def save_checkpoint(self, iteration: int):
        checkpoint = {"encoder": self.smirk_encoder.state_dict(), "generator": self.smirk_generator.state_dict()}
        checkpoint_path = self.checkpoints_dir / f"smirk_iter{iteration:04d}.pth"
        torch.save(checkpoint, checkpoint_path)

    def forward(self, views_crop: Dict, training: bool) -> EncodedValues:
        image = views_crop["image"]
        B = image.shape[0]
        outputs = self.smirk_encoder(image)

        # Convert outputs to the common representation (DECA)
        values = EncodedValues()
        values.expcode = outputs["expression_params"]
        values.posecode = torch.cat((outputs["pose_params"], outputs["jaw_params"]), -1)
        values.cam = outputs["cam"]
        values.shapecode = outputs["shape_params"][:,:100] # SMIRK uses 300 shape codes, but this makes no difference
        # dummy values for texture and light (not predicted by SMIRK)
        values.texcode = torch.zeros((B, 50), dtype=torch.float, device=self.device)
        values.lightcode = torch.zeros((B, 9, 3), dtype=torch.float, device=self.device)
        values.eyelids = outputs["eyelid_params"]
        values.images = image
        values.lmk = views_crop["landmark"]
        values.lmk_mp = views_crop["landmark_mediapipe"]
        values.masks = views_crop["mask"].view(-1, image.shape[-2], image.shape[-1])

        # if not training:
        #     # Since we're predicting the shape of the same person, we average the shape code over the batch
        #     values.shapecode = values.shapecode.mean(0, keepdim=True).expand_as(values.shapecode)

        return values

 