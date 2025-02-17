from pathlib import Path

import click
from dotenv import load_dotenv
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch

from histopathossl.models.moco_ligthing import MoCoV2Lightning
from histopathossl.training.dataset_wsi import WSIDataset, DummyDataset
from histopathossl.training.augmentations import GaussianBlur


def get_augmentations(aug_plus=True):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(224,
                                         scale=(0.9, 1.0)),  # TODO: CHECK THIS
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                p=0.8,  # not strengthened
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(
                [GaussianBlur([0.1, 2.0])],
                p=0.5,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    return transforms.Compose(augmentation)


def find_matching_mask(wsi_id, mask_dir):
    matches = list(Path(mask_dir).rglob(f"*{wsi_id}.svs_mask_use.png"))
    return matches[0] if matches else None


def infinite_dataloader(dataloader):
    """Creates an infinite dataloader by cycling through the DataLoader forever."""
    while True:
        for batch in dataloader:
            yield batch  # Yield one batch at a time


def main():
    batch_size = 256
    queue_size = 65536
    base_encoder = "convnext_large" # work with "resnet50"
    output_dim = 128
    momentum = 0.999
    temperature = 0.07
    learning_rate = 1e-3
    max_epochs = 2
    limit_train_batches = 100
    num_workers = 0
    gpu_id = 1
    enable_cudnn_benchmark = True

    # Load environment variables
    load_dotenv()
    if enable_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    project_dir = Path(__file__).parents[2].resolve()
    wsi_dir = project_dir / "data/raw/tcga_subset"
    # mask_dir = "/home/valentin/workspaces/histopathossl/data/interim/masks/output"
    mask_dir = "/home/valentin/workspaces/histolung/data/interim/masks"
    wsi_paths = list(wsi_dir.rglob("*.svs"))
    wsi_ids = [p.stem.split(".")[0] for p in wsi_paths]
    mask_paths = [
        m for m in Path(mask_dir).rglob(f"*.svs_mask_use.png")
        if m.stem.split(".")[0] in wsi_ids
    ]
    wsi_paths.sort(key=lambda x: x.stem.split(".")[0])
    mask_paths.sort(key=lambda x: x.stem.split(".")[0])
    wsi_ids.sort()

    train_dataset = WSIDataset(
        wsi_paths,
        mask_paths,
        transform=transforms.ToTensor(),
        # transform=get_augmentations(),
    )
    # train_dataset = DummyDataset(transform=transforms.ToTensor())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
        # pin_memory=True,
        # persistent_workers=True,  # Relevant at the end of the epoch
        # prefetch_factor=4,
    )
    # Convert DataLoader into an infinite iterator
    infinite_loader = infinite_dataloader(train_loader)

    # Step 6: Initialize model
    model = MoCoV2Lightning(
        base_encoder=base_encoder,
        output_dim=output_dim,
        queue_size=queue_size,
        momentum=momentum,
        temperature=temperature,
        lr=learning_rate,
    )

    # Step 7: Train model
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        precision="16-mixed",
        limit_train_batches=limit_train_batches,
        devices=[gpu_id],
    )
    trainer.fit(model, infinite_loader)


if __name__ == "__main__":
    main()
