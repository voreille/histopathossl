import random
from pathlib import Path

import click
import torch
from dotenv import load_dotenv
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from lightning.pytorch.loggers import TensorBoardLogger

from histopathossl.models.moco_ligthing import MoCoV2Lightning
from histopathossl.training.augmentations import GaussianBlur, TwoCropsTransform
from histopathossl.training.dataset import TileDataset

project_dir = Path(__file__).parents[2].resolve()


def get_augmentations(aug_plus=True):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),  # TODO: CHECK THIS
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


def get_eval_transformations():
    """
    Returns the transformations for evaluation.
    """
    return transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def split_patients(patient_ids, seed=42):
    """
    Splits the patient IDs into train, validation, and test sets.
    """
    random.seed(seed)
    random.shuffle(patient_ids)
    n_patients = len(patient_ids)

    train_split = int(n_patients * 0.7)
    val_split = int(n_patients * 0.85)

    train_patient_ids = patient_ids[:train_split]
    val_patient_ids = patient_ids[train_split:val_split]
    test_patient_ids = patient_ids[val_split:]

    return train_patient_ids, val_patient_ids, test_patient_ids


def get_patient_from_tcga_id(tcga_id):
    """
    Extracts the patient ID from the TCGA ID.
    """
    # Example TCGA ID: TCGA-XX-XXXX
    # Extract the first part (TCGA-XX)
    patient_id = "-".join(tcga_id.split("-")[:3])
    return patient_id


def get_train_val_test_tile_paths(data_dir, magnification_key=5):
    data_dir = project_dir / data_dir
    present_subdataset = [f.name for f in data_dir.iterdir() if f.is_dir()]
    print(f"Present subdatasets: {present_subdataset}")

    wsi_dir_paths = list(data_dir.glob(f"./*/{magnification_key}/*/"))
    patient_ids = set([get_patient_from_tcga_id(f.name) for f in wsi_dir_paths])
    patient_ids = list(patient_ids)
    patient_ids.sort()

    train_patient_ids, val_patient_ids, test_patient_ids = split_patients(
        patient_ids, seed=42
    )

    tile_paths = list(data_dir.glob(f"./*/{magnification_key}/*/*.jpg"))
    print(f"Total tiles: {len(tile_paths)}")

    train_tile_paths = []
    val_tile_paths = []
    test_tile_paths = []

    for tile_path in tile_paths:
        patient_id = get_patient_from_tcga_id(tile_path.parent.name)
        if patient_id in train_patient_ids:
            train_tile_paths.append(tile_path)
        elif patient_id in val_patient_ids:
            val_tile_paths.append(tile_path)
        elif patient_id in test_patient_ids:
            test_tile_paths.append(tile_path)

    train_tiles = len(train_tile_paths)
    val_tiles = len(val_tile_paths)
    test_tiles = len(test_tile_paths)
    print(
        f"Split tiles: Train: {train_tiles} ({train_tiles / len(tile_paths):.1%}), "
        f"Val: {val_tiles} ({val_tiles / len(tile_paths):.1%}), "
        f"Test: {test_tiles} ({test_tiles / len(tile_paths):.1%})"
    )

    return train_tile_paths, val_tile_paths, test_tile_paths


def get_dataloaders(data_dir, magnification_key=5, batch_size=32, num_workers=24):
    train_tile_paths, val_tile_paths, test_tile_paths = get_train_val_test_tile_paths(
        data_dir,
        magnification_key=magnification_key,
    )

    train_dataset = TileDataset(
        train_tile_paths,
        transform=TwoCropsTransform(get_augmentations()),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    val_dataset = TileDataset(
        val_tile_paths,
        transform=TwoCropsTransform(get_eval_transformations()),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    test_dataset = TileDataset(
        test_tile_paths,
        transform=TwoCropsTransform(get_eval_transformations()),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


@click.command()
@click.option(
    "--data-dir", show_default=True, default="data/tcga-ut", help="Directory for data."
)
@click.option(
    "--magnification-key", show_default=True, default=5, help="Magnification key."
)
@click.option(
    "--batch-size", default=32, show_default=True, help="Batch size for training."
)
@click.option(
    "--queue-size",
    default=4096,
    show_default=True,
    help="Queue size for negative samples, to test btwn 4096 and 8192 or else",
)
@click.option(
    "--base-encoder",
    default="resnet50",
    show_default=True,
    help="Base encoder for the MoCoV2 model.",
)
@click.option(
    "--output-dim",
    default=128,
    show_default=True,
    help="Output dimension of the model.",
)
@click.option(
    "--momentum", default=0.999, show_default=True, help="Momentum for key encoder."
)
@click.option(
    "--temperature",
    default=0.07,
    show_default=True,
    help="Temperature for contrastive loss.",
)
@click.option(
    "--learning-rate",
    default=1e-3,
    show_default=True,
    help="Learning rate for training.",
)
@click.option(
    "--max-epochs", default=10, show_default=True, help="Number of training epochs."
)
@click.option(
    "--num-workers",
    default=24,
    show_default=True,
    help="Number of workers for data loading.",
)
@click.option("--gpu-id", default=0, help="GPU ID for embedding generation.")
@click.option(
    "--enable-cudnn-benchmark",
    is_flag=True,
    default=False,
    help="Enable CuDNN benchmark mode.",
)
def main(
    data_dir,
    magnification_key,
    batch_size,
    queue_size,
    base_encoder,
    output_dim,
    momentum,
    temperature,
    learning_rate,
    max_epochs,
    num_workers,
    gpu_id,
    enable_cudnn_benchmark,
):
    # Load environment variables
    load_dotenv()
    if enable_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    train_tile_paths, val_tile_paths, test_tile_paths = get_train_val_test_tile_paths(
        data_dir,
        magnification_key=magnification_key,
    )

    train_dataset = TileDataset(
        train_tile_paths,
        transform=TwoCropsTransform(get_augmentations()),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,  # Relevant at the end of the epoch
        prefetch_factor=4,
    )

    # Step 6: Initialize model
    model = MoCoV2Lightning(
        base_encoder=base_encoder,
        output_dim=output_dim,
        queue_size=queue_size,
        momentum=momentum,
        temperature=temperature,
        logger=TensorBoardLogger("tb_logs", name="linear_probing_from_embeddings"),
        lr=learning_rate,
    )

    # Step 7: Train model
    trainer = Trainer(
        max_epochs=max_epochs, accelerator="gpu", precision="16-mixed", devices=[gpu_id]
    )
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    main()
