import sys
import random
import pandas as pd
import numpy as np
import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import csv
from tqdm import tqdm
import pytorch_lightning as pl

from configilm import util
from configilm.extra.DataSets import BENv2_DataSet

# Disable xFormers for this script
os.environ["XFORMERS_DISABLED"] = "1"


# Set random seeds for reproducibility
seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
generator = torch.Generator().manual_seed(seed)

# SOFTCON PRETRAINING band information # Added B10 as a zero-initialized channel
ALL_BANDS_S2_L2A = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B10",
    "B11",
    "B12",
]

# Band statistics: mean & std (calculated from 50k data)
S2A_MEAN = [
    752.40087073,
    884.29673756,
    1144.16202635,
    1297.47289228,
    1624.90992062,
    2194.6423161,
    2422.21248945,
    2517.76053101,
    2581.64687018,
    2645.51888987,
    0,
    2368.51236873,
    1805.06846033,
]

S2A_STD = [
    1108.02887453,
    1155.15170768,
    1183.6292542,
    1368.11351514,
    1370.265037,
    1355.55390699,
    1416.51487101,
    1474.78900051,
    1439.3086061,
    1582.28010962,
    1,
    1455.52084939,
    1343.48379601,
]

datapath = {
    "images_lmdb": "/faststorage/BigEarthNet-V2/BigEarthNet-V2-LMDB",
    "metadata_parquet": "/faststorage/BigEarthNet-V2/metadata.parquet",
    "metadata_snow_cloud_parquet": "/faststorage/BigEarthNet-V2/metadata_for_patches_with_snow_cloud_or_shadow.parquet",
}

util.MESSAGE_LEVEL = util.MessageLevel.INFO  # use INFO to see all messages


# Use Pretrainining Statistics
class NormalizeWithStats:
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32).reshape(
            -1, 1, 1
        )  # Shape: (C, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(-1, 1, 1)  # Shape: (C, 1, 1)

    def __call__(self, img):
        img_np = img.numpy().astype(np.float32)  # Ensure float32
        # Standard normalization: (x - mean) / std
        img_np = (img_np - self.mean) / self.std
        return torch.from_numpy(img_np).float()  # Explicitly convert to float32


# Use 12-channel S2A statistics directly
train_mean, train_std = np.array(S2A_MEAN), np.array(S2A_STD)
print("Using pre-calculated S2A stats for 12 Sentinel-2 bands:")
print("Train mean:", train_mean)
print("Train std:", train_std)


# Drop the first 2 SAR channels to keep only Sentinel-2 bands
class DropSARChannels:
    def __call__(self, img):
        # Drop first 2 channels (VV, VH) - keep channels 2-13 (Sentinel-2 bands)
        return img[2:, :, :]


class ZeroInitializeB10:
    """Zero-initialize the B10 layer in the input image."""

    def __call__(self, img):
        b10 = torch.zeros((1, img.shape[1], img.shape[2]))  # Shape: [1, H, W]
        img = torch.cat([img[:9, :, :], b10, img[9:, :, :]], dim=0)
        return img


# Transformation pipeline for training (with augmentations)
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize to 224x224
        DropSARChannels(),  # Drop SAR bands, keep only Sentinel-2 bands
        ZeroInitializeB10(),  # Zero-initialize the B10 layer
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.RandomVerticalFlip(),  # Random vertical flip
        transforms.RandomChoice(
            [  # Randomly apply one of the rotations
                transforms.RandomRotation(degrees=(0, 0)),  # No rotation
                transforms.RandomRotation(degrees=(90, 90)),  # Rotate 90 degrees
                transforms.RandomRotation(degrees=(180, 180)),  # Rotate 180 degrees
                transforms.RandomRotation(degrees=(270, 270)),  # Rotate 270 degrees
            ]
        ),
        transforms.RandomResizedCrop(
            size=(224, 224), scale=(0.8, 1.0)
        ),  # Random resized crop
        NormalizeWithStats(
            train_mean, train_std
        ),  # Standard normalization with S2A stats
    ]
)

# Transformation pipeline for validation (no augmentations)
val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize to 224x224
        DropSARChannels(),  # Drop SAR bands, keep only Sentinel-2 bands
        ZeroInitializeB10(),  # Zero-initialize the B10 layer
        NormalizeWithStats(
            train_mean, train_std
        ),  # Standard normalization with S2A stats
    ]
)

# --- Country-specific train/val split function ---
meta = pd.read_parquet(datapath["metadata_parquet"])


def load_country_train_val(
    country: str,
    n_samples: int,
    seed: int = None,
    train_frac: float = 0.8,
    img_size=(14, 224, 224),  # Changed from (14, 120, 120) to load all channels
    include_snowy=False,
    include_cloudy=False,
):
    """
    Samples n_samples patches from the TRAIN split of `country`,
    then does an internal 80/20 train/val split (both from the original TRAIN data).
    Returns (train_ds, val_ds).
    """
    mask = (meta.country == country) & (meta.split == "train")
    available = meta.loc[mask, "patch_id"].tolist()
    if not available:
        raise ValueError(f"No TRAIN patches found for country={country!r}")
    if n_samples > len(available):
        raise ValueError(
            f"Requested {n_samples} samples but only {len(available)} TRAIN patches available for {country!r}"
        )
    rng = random.Random(seed)
    sampled = rng.sample(available, k=n_samples)
    split_at = int(n_samples * train_frac)
    train_ids = set(sampled[:split_at])
    val_ids = set(sampled[split_at:])

    def _make_ds(keep_ids, transform):
        return BENv2_DataSet.BENv2DataSet(
            data_dirs=datapath,
            img_size=img_size,
            split="train",
            include_snowy=include_snowy,
            include_cloudy=include_cloudy,
            patch_prefilter=lambda pid: pid in keep_ids,
            transform=transform,
        )

    train_ds = _make_ds(train_ids, train_transform)
    val_ds = _make_ds(val_ids, val_transform)
    return train_ds, val_ds


# --- Usage: get train/val splits for a country ---
ds_train, ds_val = load_country_train_val("Ireland", n_samples=1000, seed=seed)

# DataLoaders for train and validation
train_loader = DataLoader(ds_train, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(ds_val, batch_size=32, shuffle=False, num_workers=4)

# --- Model setup ---
sys.path.append("/home/arne/softcon")
from models.dinov2 import vision_transformer as dinov2_vitb

model_vitb14 = dinov2_vitb.__dict__["vit_base"](
    img_size=224,
    patch_size=14,
    in_chans=13,  #
    block_chunks=0,
    init_values=1e-4,
    num_register_tokens=0,
)

# Load pretrained weights excluding patch embedding layer
ckpt_vitb14 = torch.load(
    "/faststorage/softcon/pretrained/B13_vitb14_softcon_enc.pth",
    map_location="cpu",
    weights_only=True,
)

# Load state dict
model_state = model_vitb14.state_dict()
model_vitb14.load_state_dict(model_state)

print(model_vitb14)

# Wrap with LoRA
sys.path.append("/home/arne/LoRA-ViT")
from lora import LoRA_ViT_timm

lora_model = LoRA_ViT_timm(model_vitb14, num_classes=0, r=4, alpha=16)

# Add classification head
num_classes = 19
classifier = nn.Linear(model_vitb14.embed_dim, num_classes)
lora_with_head = nn.Sequential(lora_model, classifier)

# IMPORTANT: Ensure classification head is trainable
for param in classifier.parameters():
    param.requires_grad = True

print(lora_with_head)
print(
    "\nNumber of trainable parameters:",
    sum(p.numel() for p in lora_with_head.parameters() if p.requires_grad),
)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lora_with_head.to(device)


# --- Lightning Module ---
class SoftConLightningModule(pl.LightningModule):
    def __init__(self, model, embed_dim, num_classes, lr=1e-4):
        super().__init__()
        self.model = model
        self.classifier = nn.Linear(embed_dim, num_classes)  # Use embed_dim directly
        self.criterion = nn.BCEWithLogitsLoss()
        self.lr = lr

    def forward(self, x):
        features = self.model(x)
        return self.classifier(features)

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        outputs = self(imgs)
        loss = self.criterion(outputs, labels.float())
        preds = (torch.sigmoid(outputs) > 0.5).float()
        acc = (preds == labels).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        outputs = self(imgs)
        loss = self.criterion(outputs, labels.float())
        preds = (torch.sigmoid(outputs) > 0.5).float()
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# Pass the embed_dim explicitly
embed_dim = model_vitb14.embed_dim  # Extract embed_dim from the base model
pl_model = SoftConLightningModule(
    lora_model, embed_dim=embed_dim, num_classes=19, lr=1e-4
)

# --- Trainer ---
trainer = pl.Trainer(
    max_epochs=25,
    accelerator="auto",  # Automatically use GPU if available
    log_every_n_steps=10,
    strategy="deepspeed"
    if torch.cuda.is_available()
    else "auto",  # Use DeepSpeed only if CUDA is available
    default_root_dir="/faststorage/softcon/finetuning/",
)

# --- Fit ---
trainer.fit(pl_model, train_loader, val_loader)

# --- Save model ---
torch.save(pl_model.state_dict(), "/faststorage/softcon/finetuning/ireland.pt")
