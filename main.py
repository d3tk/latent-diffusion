import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from torch.amp import GradScaler, autocast
import wandb
import json
import random
from tqdm import tqdm

from packaging import version
from omegaconf import OmegaConf
from functools import partial
from PIL import Image

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config


# python3 main.py --base configs/latent-diffusion/cin-ldm-test-small.yaml --train --gpus 0 --epochs 10 --batch_size 4 --accumulate_grad_batches 4 --log_frequency 50 --checkpoint_frequency 2 --name "imagenet_test_small" --seed 42  # For reproducibility


def seed_everything(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rank_zero_only(func):
    """Decorator to run function only on rank 0"""

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def rank_zero_info(message):
    """Print info message only on rank 0"""
    print(message)


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p", "--project", help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    # Add PyTorch-specific arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of epochs to train",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="comma-separated list of gpu ids to use",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="number of batches to accumulate gradients",
    )
    parser.add_argument(
        "--checkpoint_frequency",
        type=int,
        default=10,
        help="save checkpoint every N epochs",
    )
    parser.add_argument(
        "--log_frequency",
        type=int,
        default=10,
        help="log metrics every N steps",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="sar-diffusion",
        help="wandb entity (username or team name)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="simple-vae-testing",
        help="wandb project name",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default="test-run",
        help="wandb run name (overrides --name for wandb)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="disable wandb logging",
    )
    return parser


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[
            worker_id * split_size : (worker_id + 1) * split_size
        ]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)


class DataModuleFromConfig:
    def __init__(
        self,
        batch_size,
        train=None,
        validation=None,
        test=None,
        predict=None,
        wrap=False,
        num_workers=None,
        shuffle_test_loader=False,
        use_worker_init_fn=False,
        shuffle_val_dataloader=False,
    ):
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
        if validation is not None:
            self.dataset_configs["validation"] = validation
        if test is not None:
            self.dataset_configs["test"] = test
        if predict is not None:
            self.dataset_configs["predict"] = predict
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs
        )
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def train_dataloader(self):
        is_iterable_dataset = isinstance(
            self.datasets["train"], Txt2ImgIterableBaseDataset
        )
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False if is_iterable_dataset else True,
            worker_init_fn=init_fn,
        )

    def val_dataloader(self, shuffle=False):
        if (
            isinstance(self.datasets["validation"], Txt2ImgIterableBaseDataset)
            or self.use_worker_init_fn
        ):
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(
            self.datasets["validation"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=init_fn,
            shuffle=shuffle,
        )

    def test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(
            self.datasets["test"], Txt2ImgIterableBaseDataset
        )
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=init_fn,
            shuffle=shuffle,
        )


class SetupCallback:
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def setup(self):
        # Create logdirs and save configs
        os.makedirs(self.logdir, exist_ok=True)
        os.makedirs(self.ckptdir, exist_ok=True)
        os.makedirs(self.cfgdir, exist_ok=True)

        print("Project config")
        print(OmegaConf.to_yaml(self.config))
        OmegaConf.save(
            self.config,
            os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)),
        )

        print("Lightning config")
        print(OmegaConf.to_yaml(self.lightning_config))
        OmegaConf.save(
            OmegaConf.create({"lightning": self.lightning_config}),
            os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)),
        )


class ImageLogger:
    def __init__(
        self,
        epoch_frequency,
        max_images,
        clamp=True,
        increase_log_steps=True,
        rescale=True,
        disabled=False,
        log_on_batch_idx=False,
        log_first_step=False,
        log_images_kwargs=None,
        save_dir=None,
    ):
        self.rescale = rescale
        self.epoch_freq = epoch_frequency
        self.max_images = max_images
        self.log_steps = [2**n for n in range(int(np.log2(self.epoch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.epoch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.save_dir = save_dir

    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k, global_step, current_epoch, batch_idx
            )
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(
        self, model, batch, batch_idx, global_step, current_epoch, split="train"
    ):
        # Check if we should log based on epoch frequency
        if (
            self.check_frequency(current_epoch)
            and hasattr(model, "log_images")
            and callable(model.log_images)
            and self.max_images > 0
        ):
            is_train = model.training
            if is_train:
                model.eval()

            with torch.no_grad():
                images = model.log_images(
                    batch, split=split, verbose=False, **self.log_images_kwargs
                )

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1.0, 1.0)

            if self.save_dir:
                self.log_local(
                    self.save_dir,
                    split,
                    images,
                    global_step,
                    current_epoch,
                    batch_idx,
                )

            if is_train:
                model.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.epoch_freq) == 0 or (check_idx in self.log_steps)) and (
            check_idx > 0 or self.log_first_step
        ):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False


class CUDACallback:
    def __init__(self):
        self.start_time = None

    def on_train_epoch_start(self, device):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        self.start_time = time.time()

    def on_train_epoch_end(self, device):
        torch.cuda.synchronize(device)
        max_memory = torch.cuda.max_memory_allocated(device) / 2**20
        epoch_time = time.time() - self.start_time

        rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
        rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")


class PyTorchTrainer:
    def __init__(self, model, data, config, logdir, ckptdir, device, callbacks=None):
        self.model = model
        self.data = data
        self.config = config
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.device = device
        self.callbacks = callbacks or []
        self.global_step = 0
        self.current_epoch = 0

        # Setup optimizers - handle models with multiple optimizers
        if hasattr(model, "configure_optimizers"):
            optimizers = model.configure_optimizers()
            if isinstance(optimizers, (list, tuple)) and len(optimizers) > 0:
                if isinstance(optimizers[0], (list, tuple)):
                    # Multiple optimizers
                    self.optimizers = optimizers[0]
                    self.schedulers = optimizers[1] if len(optimizers) > 1 else []
                else:
                    # Single optimizer
                    self.optimizers = [optimizers]
                    self.schedulers = []
            else:
                # Single optimizer
                self.optimizers = [optimizers]
                self.schedulers = []
        else:
            # Default optimizer
            self.optimizers = [
                optim.AdamW(
                    self.model.parameters(), lr=model.learning_rate, weight_decay=0.01
                )
            ]
            self.schedulers = []

        # Setup schedulers
        if not self.schedulers:
            self.schedulers = [
                optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=config.get("epochs", 100)
                )
                for opt in self.optimizers
            ]

        # Setup gradient scaler for mixed precision
        self.scaler = GradScaler(device=self.device, init_scale=1024, enabled=True)

        # Setup wandb logger
        # Initialize wandb with proper configuration
        if opt.no_wandb:
            print("Wandb logging disabled by --no-wandb flag")
            self.wandb_initialized = False
        else:
            try:
                # Get wandb config from command line args or config
                wandb_project = opt.project or config.get(
                    "wandb_project", "latent-diffusion"
                )
                wandb_entity = opt.wandb_entity or config.get(
                    "wandb_entity", None
                )  # Can be None for personal account
                wandb_name = (
                    opt.wandb_name
                    or opt.name
                    or config.get("wandb_name", f"latent-diffusion-{now}")
                )

                # Check if we should resume a wandb run
                wandb_run_id = None
                if opt.resume and os.path.exists(ckpt):
                    # Try to load wandb run ID from checkpoint or logdir
                    run_id_file = os.path.join(logdir, "wandb_run_id.txt")
                    if os.path.exists(run_id_file):
                        with open(run_id_file, "r") as f:
                            wandb_run_id = f.read().strip()

                # Initialize wandb
                wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=wandb_name,
                    config=OmegaConf.to_container(config, resolve=True),
                    resume="allow" if wandb_run_id else None,
                    id=wandb_run_id,
                )

                # Save wandb run ID for future resuming
                if wandb.run.id:
                    run_id_file = os.path.join(logdir, "wandb_run_id.txt")
                    with open(run_id_file, "w") as f:
                        f.write(wandb.run.id)

                print(f"Initialized wandb run: {wandb.run.name} (ID: {wandb.run.id})")
                self.wandb_initialized = True

            except Exception as e:
                print(f"Warning: Failed to initialize wandb: {e}")
                print("Continuing without wandb logging...")
                self.wandb_initialized = False

    def safe_wandb_log(self, data):
        """Safely log data to wandb, handling cases where wandb is not initialized"""
        if hasattr(self, "wandb_initialized") and self.wandb_initialized:
            try:
                wandb.log(data)
            except Exception as e:
                print(f"Warning: Failed to log to wandb: {e}")

    def finish_wandb(self):
        """Safely finish wandb run"""
        if hasattr(self, "wandb_initialized") and self.wandb_initialized:
            try:
                wandb.finish()
                print("Wandb run finished successfully")
            except Exception as e:
                print(f"Warning: Failed to finish wandb run: {e}")

    def save_checkpoint(self, filename="last.ckpt"):
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dicts": [opt.state_dict() for opt in self.optimizers],
            "scheduler_state_dicts": [sched.state_dict() for sched in self.schedulers],
            "scaler_state_dict": self.scaler.state_dict(),
            "config": self.config,
        }
        torch.save(checkpoint, os.path.join(self.ckptdir, filename))

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path, weights_only=False, map_location=self.device
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer states
        if "optimizer_state_dicts" in checkpoint:
            for i, opt_state in enumerate(checkpoint["optimizer_state_dicts"]):
                if i < len(self.optimizers):
                    self.optimizers[i].load_state_dict(opt_state)

        # Load scheduler states
        if "scheduler_state_dicts" in checkpoint:
            for i, sched_state in enumerate(checkpoint["scheduler_state_dicts"]):
                if i < len(self.schedulers):
                    self.schedulers[i].load_state_dict(sched_state)

        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        return checkpoint

    def train_epoch(self, train_loader, accumulate_grad_batches=1):
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        # Call epoch start callbacks
        for callback in self.callbacks:
            if hasattr(callback, "on_train_epoch_start"):
                callback.on_train_epoch_start(self.device)
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch} Training")
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            if isinstance(batch, (list, tuple)):
                batch = [
                    b.to(self.device) if isinstance(b, torch.Tensor) else b
                    for b in batch
                ]
            elif isinstance(batch, dict):
                # Handle dictionary batches (like ImageNet datasets)
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
            elif isinstance(batch, torch.Tensor):
                batch = batch.to(self.device)
            else:
                # For other types, try to move to device if possible
                try:
                    batch = batch.to(self.device)
                except AttributeError:
                    # If it doesn't have .to() method, leave as is
                    pass

            # Handle multiple optimizers
            if len(self.optimizers) > 1:
                # For models with multiple optimizers (like autoencoders with discriminator)
                total_batch_loss = 0.0
                for optimizer_idx, optimizer in enumerate(self.optimizers):
                    optimizer.zero_grad()

                    # Forward pass with mixed precision
                    with autocast(
                        device_type="cuda", dtype=torch.bfloat16, enabled=True
                    ):
                        result = self.model.training_step(
                            batch, batch_idx, optimizer_idx
                        )
                        # Handle tuple return (loss, loss_dict) from training_step
                        if isinstance(result, tuple):
                            loss, loss_dict = result
                        else:
                            loss = result
                            loss_dict = {}
                        loss = loss / accumulate_grad_batches

                    # Backward pass
                    self.scaler.scale(loss).backward()
                    total_batch_loss += loss.item() * accumulate_grad_batches

                # Step all optimizers
                if (batch_idx + 1) % accumulate_grad_batches == 0:
                    for optimizer in self.optimizers:
                        self.scaler.step(optimizer)
                    self.scaler.update()
                    for optimizer in self.optimizers:
                        optimizer.zero_grad()

                total_loss += total_batch_loss
            else:
                # Single optimizer case
                # Forward pass with mixed precision
                with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    result = self.model.training_step(batch, batch_idx)
                    # Handle tuple return (loss, loss_dict) from training_step
                    if isinstance(result, tuple):
                        loss, loss_dict = result
                    else:
                        loss = result
                        loss_dict = {}
                    loss = loss / accumulate_grad_batches

                # Backward pass
                self.scaler.scale(loss).backward()

                # Gradient accumulation
                if (batch_idx + 1) % accumulate_grad_batches == 0:
                    self.scaler.step(self.optimizers[0])
                    self.scaler.update()
                    self.optimizers[0].zero_grad()

                total_loss += loss.item() * accumulate_grad_batches

            # Log metrics
            current_loss = total_loss / (batch_idx + 1)
            log_dict = {
                "train_loss": current_loss,
                "epoch": self.current_epoch,
                "global_step": self.global_step,
            }
            # Add learning rates for all optimizers
            for i, opt in enumerate(self.optimizers):
                log_dict[f"learning_rate_{i}"] = opt.param_groups[0]["lr"]
            self.safe_wandb_log(log_dict)

            self.global_step += 1
            pbar.set_postfix(
                loss=f"{current_loss:.4f}",
                lr=[opt.param_groups[0]["lr"] for opt in self.optimizers],
                gb=f"{torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB",
            )
        # Call epoch end callbacks
        for callback in self.callbacks:
            if hasattr(callback, "on_train_epoch_end"):
                callback.on_train_epoch_end(self.device)

        # Step schedulers
        for scheduler in self.schedulers:
            scheduler.step()

        return total_loss / num_batches

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)

        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(val_loader, desc=f"Epoch {self.current_epoch} Validation")
            ):
                # Move batch to device - handle different batch types
                if isinstance(batch, (list, tuple)):
                    batch = [
                        b.to(self.device) if isinstance(b, torch.Tensor) else b
                        for b in batch
                    ]
                elif isinstance(batch, dict):
                    # Handle dictionary batches (like ImageNet datasets)
                    batch = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                elif isinstance(batch, torch.Tensor):
                    batch = batch.to(self.device)
                else:
                    # For other types, try to move to device if possible
                    try:
                        batch = batch.to(self.device)
                    except AttributeError:
                        # If it doesn't have .to() method, leave as is
                        pass

                # Forward pass
                with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    result = self.model.validation_step(batch, batch_idx)
                    # Handle tuple return (loss, loss_dict) from validation_step
                    if isinstance(result, tuple):
                        loss, loss_dict = result
                    else:
                        loss = result
                        loss_dict = {}

                total_loss += loss.item()

        avg_loss = total_loss / num_batches
        self.safe_wandb_log(
            {
                "val_loss": avg_loss,
                "epoch": self.current_epoch,
                "global_step": self.global_step,
            }
        )
        return avg_loss

    def test(self, test_loader):
        self.model.eval()
        total_loss = 0.0
        num_batches = len(test_loader)

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
                # Move batch to device - handle different batch types
                if isinstance(batch, (list, tuple)):
                    batch = [
                        b.to(self.device) if isinstance(b, torch.Tensor) else b
                        for b in batch
                    ]
                elif isinstance(batch, dict):
                    # Handle dictionary batches (like ImageNet datasets)
                    batch = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                elif isinstance(batch, torch.Tensor):
                    batch = batch.to(self.device)
                else:
                    # For other types, try to move to device if possible
                    try:
                        batch = batch.to(self.device)
                    except AttributeError:
                        # If it doesn't have .to() method, leave as is
                        pass

                # Forward pass
                with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    result = self.model.test_step(batch, batch_idx)
                    # Handle tuple return (loss, loss_dict) from test_step
                    if isinstance(result, tuple):
                        loss, loss_dict = result
                    else:
                        loss = result
                        loss_dict = {}

                total_loss += loss.item()

        avg_loss = total_loss / num_batches
        self.safe_wandb_log({"test_loss": avg_loss})
        return avg_loss

    def fit(self, epochs):
        train_loader = self.data.train_dataloader()
        val_loader = (
            self.data.val_dataloader() if hasattr(self.data, "val_dataloader") else None
        )

        best_val_loss = float("inf")

        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch(
                train_loader, self.config.get("accumulate_grad_batches", 1)
            )

            # Validate
            val_loss = float("inf")
            if val_loader is not None:
                val_loss = self.validate(val_loader)

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint("best.ckpt")

            # Save checkpoint periodically
            if epoch % self.config.get("checkpoint_frequency", 10) == 0:
                self.save_checkpoint(f"epoch_{epoch:06}.ckpt")

            # Save last checkpoint
            self.save_checkpoint("last.ckpt")

            # Call image logging callbacks at the end of each epoch
            for callback in self.callbacks:
                if hasattr(callback, "log_img"):
                    # Get a sample batch for image logging
                    sample_batch = next(iter(train_loader))
                    # Move batch to device
                    if isinstance(sample_batch, (list, tuple)):
                        sample_batch = [
                            b.to(self.device) if isinstance(b, torch.Tensor) else b
                            for b in sample_batch
                        ]
                    elif isinstance(sample_batch, dict):
                        sample_batch = {
                            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in sample_batch.items()
                        }
                    elif isinstance(sample_batch, torch.Tensor):
                        sample_batch = sample_batch.to(self.device)

                    callback.log_img(
                        self.model,
                        sample_batch,
                        0,  # batch_idx
                        self.global_step,
                        self.current_epoch,
                        "train",
                    )

            print(
                f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

        self.finish_wandb()


if __name__ == "__main__":
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()

    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        # print the config to verify
        # print(f"before override: {config}")
        # Add command line arguments to config
        config.epochs = opt.epochs
        config.gpus = opt.gpus
        config.accumulate_grad_batches = opt.accumulate_grad_batches
        config.checkpoint_frequency = opt.checkpoint_frequency
        config.log_frequency = opt.log_frequency
        # print(f"after override: {config}")
        # Setup device
        if torch.cuda.is_available() and opt.gpus:
            device_ids = [int(x.strip()) for x in opt.gpus.split(",")]
            device = torch.device(f"cuda:{device_ids[0]}")
            if len(device_ids) > 1:
                print(f"Using multiple GPUs: {device_ids}")
        else:
            device = torch.device("cpu")
            print("Using CPU")

        # model
        model = instantiate_from_config(config.model)
        model = model.to(device)
        model.device = device  # Add device attribute for compatibility

        # Fix device mismatch for logvar tensor if it exists
        if hasattr(model, "logvar") and model.logvar.device != device:
            model.logvar = model.logvar.to(device)
            if model.learn_logvar:
                # If logvar is a parameter, we need to update the parameter reference
                for name, param in model.named_parameters():
                    if name == "logvar":
                        param.data = model.logvar.data
                        break

        # data
        data = instantiate_from_config(config.data)
        # NOTE calling these ourselves should not be necessary but it is.
        data.prepare_data()
        data.setup()
        print("#### Data #####")
        for k in data.datasets:
            print(
                f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}"
            )

        # configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        if device.type == "cuda":
            ngpu = len(device_ids) if len(device_ids) > 1 else 1
        else:
            ngpu = 1
        accumulate_grad_batches = config.accumulate_grad_batches
        # print(f"accumulate_grad_batches = {accumulate_grad_batches}")

        if opt.scale_lr:
            model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            print(
                "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr
                )
            )
        else:
            model.learning_rate = base_lr
            print("++++ NOT USING LR SCALING ++++")
            print(f"Setting learning rate to {model.learning_rate:.2e}")

        # Setup callbacks
        callbacks = []

        # Setup callback
        setup_callback = SetupCallback(
            opt.resume, now, logdir, ckptdir, cfgdir, config, OmegaConf.create()
        )
        setup_callback.setup()

        # Image logger callback
        image_logger = ImageLogger(
            epoch_frequency=5, max_images=4, clamp=True, save_dir=logdir
        )
        callbacks.append(image_logger)

        # CUDA callback
        cuda_callback = CUDACallback()
        callbacks.append(cuda_callback)

        # Create trainer
        trainer = PyTorchTrainer(
            model=model,
            data=data,
            config=config,
            logdir=logdir,
            ckptdir=ckptdir,
            device=device,
            callbacks=callbacks,
        )

        # Load checkpoint if resuming
        if opt.resume and os.path.exists(ckpt):
            print(f"Loading checkpoint from {ckpt}")
            trainer.load_checkpoint(ckpt)

        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            print("Summoning checkpoint.")
            trainer.save_checkpoint()

        def divein(*args, **kwargs):
            import pudb

            pudb.set_trace()

        import signal

        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train:
            try:
                trainer.fit(config.epochs)
            except Exception:
                melk()
                raise
        if not opt.no_test:
            test_loader = data.test_dataloader()
            trainer.test(test_loader)

    except Exception:
        if opt.debug:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # Ensure wandb is properly finished
        try:
            if "trainer" in locals() and hasattr(trainer, "finish_wandb"):
                trainer.finish_wandb()
            elif hasattr(wandb, "run") and wandb.run is not None:
                wandb.finish()
        except Exception as e:
            print(f"Warning: Failed to finish wandb in cleanup: {e}")

        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
