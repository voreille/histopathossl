import os
from pathlib import Path
import random

from dotenv import load_dotenv
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from histopathossl.training.dataset import TileDataset, TwoCropsTransform
from histopathossl.models.moco_ligthing import MoCoV2Lightning
from histopathossl.training.train_moco import get_augmentations

load_dotenv()

dataset_path = os.getenv("DATA_TILE_672_PATH")

tile_paths = [
    f for f in Path(dataset_path).rglob("*.png") if f.parent.stem == "tiles"
]
print(f"Total tiles found by rglob: {len(tile_paths)}")

dataset = TileDataset(
    tile_paths,
    transform=TwoCropsTransform(get_augmentations()),
)
dataloader = DataLoader(dataset, batch_size=32, num_workers=32, shuffle=True)

for batch_idx, (x_1, x_2) in enumerate(dataloader):
    print(f"Batch {batch_idx + 1}:")
    print(f"shape of x_1: {x_1.shape}")
    print(f"shape of x_2: {x_2.shape}")
    print()
    if batch_idx == 5:
        break
