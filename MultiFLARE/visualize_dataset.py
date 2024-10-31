from pathlib import Path

import torch
from tqdm import tqdm

from arguments import config_parser
from utils.dataset import DeviceDataLoader
from utils.visualization import save_img
from Avatar import Avatar

@torch.no_grad()
def main(avatar: Avatar):   
    out_dir = Path("/bulk/tmp_test/dataset")
    out_dir.mkdir(parents=True, exist_ok=True)

    device, dataset_train = avatar.device, avatar.dataset_train

    #################### Visualize dataset images ####################
    dataloader = DeviceDataLoader(dataset_train, device=device, batch_size=16, collate_fn=dataset_train.collate, num_workers=8)
    for views in tqdm(dataloader):
        img = views["img"]
        for i,idx in enumerate(views["idx"]):
            save_img(out_dir / f"{idx}.png", img[i], to_srgb=True)

if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()

    args.deformer_pretrain = 0
    args.initial_expr = "zeros"

    main(Avatar(args))
