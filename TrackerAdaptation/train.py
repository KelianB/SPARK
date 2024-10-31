import os
import time
import logging
from tqdm import tqdm
import math
from pathlib import Path
from typing import Dict

from argparse import Namespace
import torch
from torch.utils.data import Subset, Dataset, DataLoader, WeightedRandomSampler
import wandb

from main import spark_setup, spark_config_parser, parse_args
from adapt.wrapper import FaceTrackerWrapper

# MultiFLARE imports
from utils.dataset import DeviceDataLoader, find_collate
from flame.FLAME import MEDIAPIPE_LMK_EMBEDDING_INDICES

def adapt_tracker(
        args: Namespace,
        wrapper: FaceTrackerWrapper,
        dataset_train: Dataset,
        dataset_val: Dataset,
        dataset_test: Dataset,
        ):

    device = wrapper.device
    out_dir = args.out_dir
    use_wandb = bool(args.wandb_workspace)

    # Setup logging
    if use_wandb:
        wandb_path = out_dir / "wandb_logs"
        wandb_path.mkdir(parents=True, exist_ok=True)
        os.environ["WANDB_DIR"] = str(wandb_path)
        wandb.init(project=args.wandb_workspace, name=args.exp_name)

    importance_sampling = args.importance_sampling_seq or args.importance_sampling_pose or args.importance_sampling_expr or args.importance_sampling_loss_lmk

    # Pre-shuffle the training and validation sets
    torch.manual_seed(1234) # ensure we always get the same permutation
    dataset_val = Subset(dataset_val, torch.randperm(len(dataset_val)))
    dataset_test = Subset(dataset_test, torch.randperm(len(dataset_test)))
    dataset_test = Subset(dataset_test, range(len(dataset_test))[:64]) # limit the test set size

    # Setup dataloaders - we don't use workers because the dataset is cached.
    # In this case, the overhead of workers can decrease performance and we don't want to store multiple copies in RAM.
    dataloader_kwargs = {"device": device, "drop_last": False, "num_workers": 0, "pin_memory": True}
    if importance_sampling:
        weights, ordered_views_idx = compute_importance_sampling_weights(dataset_train, wrapper, device, args.importance_sampling_seq, args.importance_sampling_pose, args.importance_sampling_expr)
        sampler_weights_initial = weights.clone()
        sampler = WeightedRandomSampler(weights, len(dataset_train), replacement=True)
        dataloader_train = DeviceDataLoader(dataset_train, **dataloader_kwargs, collate_fn=find_collate(dataset_train), batch_size=args.batch_size, sampler=sampler)
    else:
        dataloader_train = DeviceDataLoader(dataset_train, **dataloader_kwargs, collate_fn=find_collate(dataset_train), batch_size=args.batch_size, shuffle=True)
    dataloader_val = DeviceDataLoader(dataset_val, **dataloader_kwargs, collate_fn=find_collate(dataset_val), batch_size=8, shuffle=False)
    dataloader_test = DeviceDataLoader(dataset_test, **dataloader_kwargs, collate_fn=find_collate(dataset_test), batch_size=8, shuffle=False)

    vis_dir_val = out_dir / "val"
    vis_dir_test = out_dir / "test"
    vis_dir_val.mkdir(parents=True, exist_ok=True)
    vis_dir_test.mkdir(parents=True, exist_ok=True)

    def test_fn(loader: DataLoader, iteration: int, vis_dir: Path,
                txt: str, log_prefix: str, wandb_data=None):
        all_losses = dict()
        n_samples = 0
        for i, views in enumerate(loader):
            vispath = (vis_dir / f"{iteration}.png") if i == 0 else None
            run_dict = wrapper.forward(views, training=False, vispath=vispath, losses=True)
            losses = run_dict["losses"]
            bs = views["crop"]["landmark"].shape[0]
            n_samples += bs
            for key in losses:
                all_losses[key] = all_losses.get(key, 0) + losses[key].detach() * bs
        all_losses = dict([(key, all_losses[key] / n_samples) for key in all_losses])
        logging.info(f"{txt}: " + " - ".join(f"{key}: {all_losses[key]:.06f}" for key in all_losses))
        if wandb_data and use_wandb:
            format_losses_metrics(all_losses, wandb_data, log_prefix)

    to_train = wrapper.encoder.get_trainables(args)

    if args.weight_reg > 0:
        original_params = [[p.clone() for p in mod.parameters()] for mod in to_train]   

    for v in to_train:
        v.requires_grad_(True)
    optimizer = torch.optim.Adam(sum([list(v.parameters()) for v in to_train], []), lr=args.adapt_lr)

    # ==============================================================================================
    # TRAINING
    # ==============================================================================================
    epochs = math.ceil(args.adapt_iters / len(dataloader_train))
    iteration = args.tracker_resume
    last_iteration = args.tracker_resume + args.adapt_iters

    print("=="*50)
    logging.info(f"Training from iteration {iteration+1} to {last_iteration} (1 epoch = {len(dataloader_train)} iters)")
    print("=="*50)

    progress_bar = tqdm(range(epochs))
    start = time.time()
    for epoch in progress_bar:
        lmk_loss_per_view = dict()

        for views in dataloader_train:
            if iteration >= last_iteration:
                break
            iteration += 1
            progress_bar.set_description(desc=f'Epoch {epoch+1}, Iter {iteration}')
            is_validate_iter = args.val_frequency > 0 and (iteration == 1 or iteration % args.val_frequency == 0)
            is_test_iter = args.test_frequency > 0 and (iteration == 1 or iteration % args.test_frequency == 0)
            is_save_iter = iteration % args.save_frequency == 0

            run_dict = wrapper(views, training=True, losses=True)
            losses_and_metrics = run_dict["losses"]
            loss = losses_and_metrics["loss"]

            if args.importance_sampling_loss_lmk:
                predicted_landmarks_mediapipe = run_dict["values"]["predicted_landmarks_mediapipe"]
                lmk_mp = run_dict["values"]["lmk_mp"]
                lmk_loss = (predicted_landmarks_mediapipe[..., :2] - lmk_mp[..., MEDIAPIPE_LMK_EMBEDDING_INDICES, :2]).abs().mean((1,2))
                for i, vidx in enumerate(views["idx"]):
                    lmk_loss_per_view[vidx.item()] = lmk_loss[i].item()

            if args.weight_reg > 0:
                for i,mod in enumerate(to_train):
                    for param, original_param in zip(mod.parameters(), original_params[i]):
                        loss += args.weight_reg * (param - original_param).pow(2).sum()

            # ==============================================================================================
            # OPTIMIZER STEP
            # ==============================================================================================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ==============================================================================================
            # VISUALIZATIONS, LOGGING AND VALIDATION
            # ==============================================================================================
            progress_bar.set_postfix({"loss": loss.detach().cpu().item()})

            wandb_data = None
            if use_wandb:
                wandb_data = {}
                format_losses_metrics(losses_and_metrics, wandb_data)

            if is_validate_iter:
                test_fn(dataloader_val, iteration, vis_dir_val, "Validation (left out frames)", "val", wandb_data)
            if is_test_iter:
                test_fn(dataloader_test, iteration, vis_dir_test, "Testing (left out sequences)", "test", wandb_data)
            
            if use_wandb:
                wandb.log(wandb_data)

            ## ============== save intermediate ==============================
            if is_save_iter:
                wrapper.encoder.save_checkpoint(iteration)

        #logging.info(f"Training (end of epoch {epoch}): " + " - ".join(f"{key}: {v/n_samples_train:.06f}" for key, v in all_losses_train.items()))

        # Update landmarks importance sampling weights
        if args.importance_sampling_loss_lmk:
            avg_lmk_loss = sum(lmk_loss_per_view.values()) / len(lmk_loss_per_view.values())
            for i, vidx in enumerate(ordered_views_idx):
                sampler.weights[i] = sampler_weights_initial[i] * (lmk_loss_per_view[vidx] if vidx in lmk_loss_per_view else avg_lmk_loss)

        

    end = time.time()
    logging.info(f"Done! Time taken: {time.strftime('%H:%M:%S', time.gmtime(end - start))}")

def compute_importance_sampling_weights(dataset_train, wrapper, device,
                                        importance_sampling_sequences: bool,
                                        importance_sampling_pose: bool,
                                        importance_sampling_expr: bool):
    dataloader_train = DeviceDataLoader(dataset_train, device=device, num_workers=0, collate_fn=find_collate(dataset_train), batch_size=args.batch_size)#, pin_memory=True)
    weights_all = torch.tensor([], dtype=torch.float, device=device)
    logging.info("Preparing importance sampling")

    frames_per_sequence = dict()
    pose_all = dict()
    expr_all = dict()
    
    logging.info("(1/2) Computing statistics")
    for views in tqdm(dataloader_train):
        values = wrapper.encode(views, training=False)
        pose, expr = values["posecode"], values["expcode"]

        for sidx in views["seq_idx"]:
            sidx = sidx.item()
            if sidx not in frames_per_sequence:
                frames_per_sequence[sidx] = 0
            frames_per_sequence[sidx] += 1
        
        for i,vidx in enumerate(views["idx"]):
            vidx = vidx.item()
            pose_all[vidx] = pose[i]
            expr_all[vidx] = expr[i]
    
    from sklearn.cluster import KMeans
    import numpy as np

    def cluster(cluster_features: torch.Tensor, n_clusters: int):
        # Use KMeans to cluster the features
        cluster_features = (cluster_features - cluster_features.mean(0)) / (1e-8 + cluster_features.std(0))

        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
        kmeans.fit(cluster_features.cpu().numpy())
        cluster_centers = torch.from_numpy(kmeans.cluster_centers_).to(cluster_features.device)
        cluster_indices = []
        for k in range(n_clusters):
            # Select all items in the cluster
            cluster_items_idx = torch.from_numpy(np.where(kmeans.labels_ == k)[0]).to(cluster_features.device)
            # Sort within the cluster by distance
            # distances = torch.linalg.vector_norm(cluster_features[cluster_items_idx] - cluster_centers[k], dim=-1)
            # cluster_items_idx = cluster_items_idx[distances.sort().indices]
            cluster_indices.append(cluster_items_idx)
        return cluster_indices, cluster_centers

    i_to_vidx = list(pose_all.keys())
    pose_all = torch.stack(list(pose_all.values()))
    expr_all = torch.stack(list(expr_all.values()))

    if importance_sampling_pose:
        # use only the global rotation
        """
        pose_features = pose_all[...,:3]
        pose_cluster_indices, _ = cluster(pose_features, 5)
        """
        for i in range(3):
            print(f"global rotation #{i}")
            vals = pose_all[..., i]
            print(f"min = {vals.min()*180/math.pi:.2f} deg, max = {vals.max()*180/math.pi:.2f} deg")
            print(f"q0.2 = {vals.quantile(0.2)*180/math.pi:.2f} deg, q0.8 = {vals.quantile(0.8)*180/math.pi:.2f} deg")

        # intervals = [(-180, -45), (-45, -30), (-30, -15), (-15, 0), (0, 15), (15, 30), (30, 45), (45, 180)]
        intervals = [(-180, -45), (-45, -30), (-30, -15), (-15, 15), (15, 30), (30, 45), (45, 180)]
        pose_cluster_indices = [[] for _ in intervals]
        for i,vidx in enumerate(i_to_vidx):
            angle = pose_all[i, 1] * 180 / math.pi
            for k, (mini, maxi) in enumerate(intervals):
                if mini < angle < maxi:
                    pose_cluster_indices[k].append(i)
                    break

        logging.info(f"Size of pose clusters: {[len(indices) for indices in pose_cluster_indices]}")
        pose_cluster_sizes = dict()
        for indices in pose_cluster_indices:
            cluster_size = len(indices)
            for i in indices:
                pose_cluster_sizes[i_to_vidx[i]] = cluster_size

    if importance_sampling_expr:
        # concatenate the 10 first expression coefficients with the jaw pose
        expr_features = torch.cat((expr_all[..., :10], pose_all[:, 3:]), dim=-1)
        expr_cluster_indices, _ = cluster(expr_features, 10)
        logging.info(f"Size of expr clusters: {[len(indices) for indices in expr_cluster_indices]}")
        expr_cluster_sizes = dict()
        for indices in expr_cluster_indices:
            cluster_size = len(indices)
            for i in indices:
                expr_cluster_sizes[i_to_vidx[i]] = cluster_size


    logging.info("(2/2) Computing weights")
    for views in tqdm(dataloader_train):
        B = views["idx"].shape[0]
        # run_dict = run_fn(views, None, training=False, losses=False)
        # losses_and_metrics = run_dict["losses"]
        # loss = losses_and_metrics["loss"]

        weights_sequence = torch.ones((B), dtype=torch.float, device=device)
        weights_pose = torch.ones_like(weights_sequence)
        weights_expr = torch.ones_like(weights_sequence)

        if importance_sampling_sequences:
            weights_sequence = 1 / torch.tensor([frames_per_sequence[sidx.item()] for sidx in views["seq_idx"]], dtype=torch.float, device=device)

        if importance_sampling_pose:
            weights_pose = 1 / torch.tensor([pose_cluster_sizes[vidx.item()] for vidx in views["idx"]], dtype=torch.float, device=device)

        if importance_sampling_expr:
            weights_expr = 1 / torch.tensor([expr_cluster_sizes[vidx.item()] for vidx in views["idx"]], dtype=torch.float, device=device)

        # weights = weights_sequence * weights_pose * weights_expr
        weights = weights_sequence * weights_pose * weights_expr

        # for i,vidx in enumerate(views["idx"]):
        #     weights_all[vidx.item()] = weights[i]
        weights_all = torch.cat((weights_all, weights), dim=0)

    print(weights_all)

    return weights_all, i_to_vidx

def format_losses_metrics(losses_and_metrics: Dict, log_data: Dict, prefix=""):
    for key in losses_and_metrics:
        key_formatted = key.replace("metric_", "").replace("loss_", "")
        if prefix:
            key_formatted = f"{prefix}_{key_formatted}"
        log_data[key_formatted] = losses_and_metrics[key]


if __name__ == "__main__":
    parser = spark_config_parser()
    # Add training-specific args
    parser.add_argument("--wandb_workspace", type=str, help="Weights & Biases workspace")
    parser.add_argument("--adapt_iters", type=int, default=3000, help="Number of iterations for adapting the tracker")
    parser.add_argument("--adapt_lr", type=float, default=5e-5, help="Learning rate for adapting the tracker")
    parser.add_argument("--val_frequency", type=int, default=200, help="Frequency of validation on left-out frames (in iterations)")
    parser.add_argument("--test_frequency", type=int, default=200, help="Frequency of test on unseen sequences (in iterations)")
    parser.add_argument("--save_frequency", type=int, default=1000, help="Frequency for saving checkpoints (in iterations)")
    parser.add_argument("--train_backbones", action="store_true", help="Train backbones")
    parser.add_argument("--train_backbones_last", action="store_true", help="Train the last conv layer of the backbone")
    parser.add_argument("--train_mlps", action="store_true", help="Train MLPs")
    parser.add_argument("--importance_sampling_seq", action="store_true", help="Use importance sampling for sequences")
    parser.add_argument("--importance_sampling_pose", action="store_true", help="Use importance sampling for poses")
    parser.add_argument("--importance_sampling_expr", action="store_true", help="Use importance sampling for expressions")
    parser.add_argument("--importance_sampling_loss_lmk", action="store_true", help="Use importance sampling on landmarks loss")
    parser.add_argument("--weight_reg", type=float, default=0, help="Weight of the loss for regularizing the parameters of the model")

    args = parse_args(parser)

    wrapper, dataset_train, dataset_val, dataset_test = \
        spark_setup(args, render_mode="crop", training=True)

    adapt_tracker(args, wrapper, dataset_train, dataset_val, dataset_test)
