import os
from pathlib import Path

import click
from dotenv import load_dotenv
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, transforms
import torch

from histopathossl.models.moco_ligthing import MoCoV2Lightning
from histopathossl.training.dataset import SuperpixelMoCoDataset
from histopathossl.training.augmentations import GaussianBlur, TwoCropsTransform


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


@click.command()
@click.option("--batch-size",
              default=32,
              show_default=True,
              help="Batch size for training.")
@click.option(
    "--queue-size",
    default=65536,
    show_default=True,
    help="Queue size for negative samples.",
)
@click.option(
    "--base-encoder",
    default="resnet50",
    show_default=True,
    help="Base encoder for the MoCoV2 model.",
)
@click.option("--output-dim",
              default=128,
              show_default=True,
              help="Output dimension of the model.")
@click.option("--momentum",
              default=0.999,
              show_default=True,
              help="Momentum for key encoder.")
@click.option(
    "--temperature",
    default=0.07,
    show_default=True,
    help="Temperature for contrastive loss.",
)
@click.option("--learning-rate",
              default=1e-3,
              show_default=True,
              help="Learning rate for training.")
@click.option("--max-epochs",
              default=10,
              show_default=True,
              help="Number of training epochs.")
@click.option("--num-workers",
              default=24,
              show_default=True,
              help="Number of workers for data loading.")
@click.option('--gpu-id', default=0, help='GPU ID for embedding generation.')
@click.option('--enable-cudnn-benchmark',
              is_flag=True,
              default=False,
              help='Enable CuDNN benchmark mode.')
@click.option(
    '--superpixel-tile-map',
    type=click.Path(),
    default=
    "/home/valentin/workspaces/histolung/data/interim/tiles_superpixels/training_superpixel_tile_map.json",
    help='path to the *.json mapping superpixels to tile paths')
def main(batch_size, queue_size, base_encoder, output_dim, momentum,
         temperature, learning_rate, max_epochs, num_workers, gpu_id,
         enable_cudnn_benchmark, superpixel_tile_map):
    # Load environment variables
    load_dotenv()
    if enable_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    project_dir = Path(__file__).parents[2].resolve()
    superpixel_tile_map = project_dir / superpixel_tile_map

    train_dataset = SuperpixelMoCoDataset(
        superpixel_tile_map,
        transform=get_augmentations(),
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
        lr=learning_rate,
    )

    # Step 7: Train model
    trainer = Trainer(max_epochs=max_epochs,
                      accelerator="gpu",
                      precision="16-mixed",
                      devices=[gpu_id])
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    main()
