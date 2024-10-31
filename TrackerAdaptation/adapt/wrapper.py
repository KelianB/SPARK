from typing import Dict

from configargparse import Namespace
import torch
from torch import Tensor

# MultiFLARE imports
from Avatar import Avatar as MultiFLARE, setup_logging

from adapt.face_encoder import FaceEncoder, DECAEncoder, SMIRKEncoder
from adapt.face_decoder import FaceDecoder, DECADecoder, SMIRKDecoder, MultiFLAREDecoder
from adapt.common_types import EncodedValues, DecodedValues
from adapt.deca_helpers import make_outputs_full_sized, create_deca
from adapt.general_utils import working_dir
from adapt.constants import MULTIFLARE_PATH

class FaceTrackerWrapper(torch.nn.Module):
    def __init__(self, args: Namespace, render_mode: str, training: bool):
        super().__init__()

        # Setup
        setup_logging()
        args.out_dir = args.multiflare_args.output_dir / args.multiflare_args.run_name / args.exp_name
        args.out_dir.mkdir(parents=True, exist_ok=True)
        device = torch.device("cpu")
        if torch.cuda.is_available() and args.device >= 0:
            device = torch.device(f"cuda:{args.device}")
        self.device = device

        self.encoder = FaceTrackerWrapper._create_encoder(args, device, training)
        self.decoder = FaceTrackerWrapper._create_decoder(args, device, render_mode, self.encoder)

        # always use eval() mode, this is what performs best for DECA at least
        self.encoder.set_train_mode(False)


    def _create_encoder(args: Namespace, device: torch.device, training: bool) -> FaceEncoder:
        checkpoints_dir = args.out_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        inference_only = not training
        if args.encoder == "DECA":
            encoder = DECAEncoder(device, checkpoints_dir, args.deca_model, args.deca_cfg, args.tracker_resume, args.deca_detail, inference_only=inference_only)
        elif args.encoder == "SMIRK":
            encoder = SMIRKEncoder(device, checkpoints_dir, args.tracker_resume, inference_only=inference_only)
        else:
            raise NotImplementedError(f"Unsupported encoder '{args.encoder}'")
        return encoder

    def _create_decoder(args: Namespace, device: torch.device, render_mode: str, encoder: FaceEncoder) -> FaceDecoder:   
        if isinstance(encoder, DECAEncoder):
            deca = encoder.deca
        else:
            # Create a dummy DECA for the decoder (it needs it for computing losses)
            # Note that the specific DECA model we're using impacts the loss weights!
            deca = create_deca("EMOCA_v2_lr_mse_20", "cfg_baseline.yaml", detail_mode=False, inference_only=False).to(device)

        if args.decoder == "DECA":
            decoder = DECADecoder(device, render_mode, deca)
        elif args.decoder == "SMIRK":
            if args.encoder != "SMIRK": raise ValueError("SMIRK decoder only works with SMIRK encoder")
            decoder = SMIRKDecoder(device, render_mode, deca, encoder.smirk_generator)
        elif args.decoder == "MultiFLARE":
            with working_dir(MULTIFLARE_PATH):
                multiflare = MultiFLARE(args.multiflare_args)
            decoder = MultiFLAREDecoder(device, render_mode, deca, multiflare)
        else:
            raise NotImplementedError(f"Unsupported decoder '{args.decoder}'")
        return decoder

    def encode(self, views: Dict, training: bool) -> EncodedValues:
        with torch.set_grad_enabled(training):
            encoded: EncodedValues = self.encoder(views["crop"], training)

            # Modify encoded values to render on the full original view instead of the crop
            if self.decoder.render_mode == "full":
                make_outputs_full_sized(encoded, views)

        return encoded

    def decode(self, encoded: EncodedValues, views: Dict, training: bool) -> DecodedValues:
        with torch.set_grad_enabled(training):
            decoded = self.decoder(encoded, views, training)
        return decoded

    def forward(self, views: Dict, training: bool, losses=False, visdict=False, vispath=None):
        with torch.set_grad_enabled(training):
            encoded: EncodedValues = self.encode(views, training)
            decoded: DecodedValues = self.decode(encoded, views, training)

            out = {"values": decoded}
            if losses:
                out["losses"] = self.decoder.loss(decoded, views["crop"], training=training)
            if visdict or vispath is not None:
                out["visdict"] = self.decoder.visualize(decoded, save_path=vispath)
        return out

    def rasterize(self, decoded_values: DecodedValues, scale: int = 1):
        return self.decoder.rasterize(decoded_values, resolution=self.decoder.resolution*scale)
        
    def rasterize_semantic_masks(self, decoded_values: DecodedValues, semantic_keys, rast=None):
        device = self.device
        B, _, H, W = decoded_values["images"].shape
        if not semantic_keys:
            return torch.zeros((B, 1, H, W), dtype=torch.float, device=device)

        if rast is None:
            rast = self.rasterize(decoded_values)

        bi, bj, bk, _, triangle_ids = rast.flatten(1, 2).unbind(-1) # (B, H*W)
        # if triangle_id is zero, then no triangle was rendered here, otherwise we need to offset the ID by one
        triangle_ids = triangle_ids.long() - 1 # (B, W*H)

        vert_masks = dict((key, self.decoder.get_vertex_mask(key)) for key in semantic_keys)

        pixel_masks = dict()
        for key in semantic_keys:
            triangle_mask = torch.stack([vert_masks[key][v] for v in self.decoder.faces[triangle_ids]]) # (B, H*W, 3)
            pixel_mask = bi * triangle_mask[...,0] + bj * triangle_mask[...,1] + bk * triangle_mask[...,2] # (B, H*W)
            pixel_mask = pixel_mask.unflatten(1, (H,W)).unsqueeze(1) # (B, 1, H, W)
            pixel_masks[key] = pixel_mask

        return pixel_masks
