from pathlib import Path
import random


def list_wsi_paths(dataset_path):
    """
    List all WSI directories in the dataset.

    Args:
        dataset_path (str or Path): Root path of the dataset.

    Returns:
        list: List of WSI directory paths.
    """
    dataset_path = Path(dataset_path)
    wsi_paths = [
        wsi_dir for wsi_dir in dataset_path.glob("*/*/") if wsi_dir.is_dir()
    ]
    return wsi_paths


def split_wsi_paths(wsi_paths, val_ratio=0.1, seed=42):
    """
    Split the WSI paths into training and validation sets.

    Args:
        wsi_paths (list): List of WSI directory paths.
        val_ratio (float): Proportion of WSI paths to use for validation.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (train_wsi_paths, val_wsi_paths)
    """
    random.seed(seed)
    random.shuffle(wsi_paths)
    split_idx = int(len(wsi_paths) * (1 - val_ratio))
    train_wsi_paths = wsi_paths[:split_idx]
    val_wsi_paths = wsi_paths[split_idx:]
    return train_wsi_paths, val_wsi_paths


def aggregate_tile_paths(wsi_paths):
    """
    Aggregate all tile paths from a list of WSI directories.

    Args:
        wsi_paths (list): List of WSI directory paths.

    Returns:
        list: List of all tile image paths.
    """
    tile_paths = []
    for wsi_path in wsi_paths:
        tiles_dir = wsi_path / "tiles"
        tile_paths.extend(tiles_dir.glob("*.png"))
    return tile_paths
