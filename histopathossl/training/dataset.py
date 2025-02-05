import json
import random
from pathlib import Path

from torch.utils.data import Dataset
# import pyspng
from PIL import Image


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class TileDataset(Dataset):

    def __init__(self, tile_paths, transform=None):
        """
        Tile-level dataset that returns individual tile images from a list of paths.

        Args:
            tile_paths (list): List of paths to tile images for a WSI.
            transform (callable, optional): Transform to apply to each tile image.
        """
        self.tile_paths = tile_paths
        self.transform = transform

    def __len__(self):
        return len(self.tile_paths)

    def __getitem__(self, idx):
        tile_path = self.tile_paths[idx]
        image = Image.open(tile_path).convert("RGB")  # Load as PIL image

        if self.transform:
            image = self.transform(image)  # Apply augmentation

        return image


class SuperpixelMoCoDataset(Dataset):
    """
    A dataset that draws two random tiles from the same superpixel and applies MoCo transformations.

    Args:
        mapping_json (str): Path to the JSON file containing the mapping.
        transform (callable, optional): MoCo-style augmentation transform.
    """

    def __init__(self, mapping_json, transform=None):
        # Load the mapping as a list of dictionaries
        with open(mapping_json, "r") as f:
            self.superpixel_list = json.load(f)

        self.transform = transform

    def __len__(self):
        return len(self.superpixel_list)

    def __getitem__(self, idx):
        """
        Returns two different tiles belonging to the same superpixel.

        Returns:
            (Tensor, Tensor): Two augmented images.
        """
        superpixel_data = self.superpixel_list[idx]
        tile_paths = superpixel_data["tile_paths"]

        # Sample two tiles (with replacement)
        tile_path_1 = random.choice(tile_paths)
        tile_path_2 = random.choice(tile_paths)

        # Load images
        image_1 = Image.open(tile_path_1).convert("RGB")
        image_2 = Image.open(tile_path_2).convert("RGB")

        # Apply MoCo transformations
        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)

        return image_1, image_2


class SuperpixelMoCoDatasetDebug(Dataset):
    """
    A dataset that draws a configurable number of random tiles from the same superpixel 
    and applies MoCo transformations.

    Args:
        mapping_json (str): Path to the JSON file containing the mapping.
        num_tiles (int): Number of tiles to return per sample.
        transform (callable, optional): MoCo-style augmentation transform.
    """

    def __init__(self, mapping_json, num_tiles=2, transform=None):
        assert num_tiles > 0, "num_tiles must be at least 1"

        # Load the mapping as a list of dictionaries
        with open(mapping_json, "r") as f:
            self.superpixel_list = json.load(f)

        self.num_tiles = num_tiles
        self.transform = transform

    def __len__(self):
        return len(self.superpixel_list)

    def __getitem__(self, idx):
        """
        Returns a specified number of tiles belonging to the same superpixel.

        Returns:
            List of transformed images.
        """
        superpixel_data = self.superpixel_list[idx]
        tile_paths = superpixel_data["tile_paths"]
        nb_tiles = len(tile_paths)

        # Sample `num_tiles` tiles (with replacement)
        sampled_tile_paths = random.choices(tile_paths, k=self.num_tiles)

        # Load images and apply transformations
        images = [
            Image.open(path).convert("RGB") for path in sampled_tile_paths
        ]

        if self.transform:
            images = [self.transform(img) for img in images]

        tile_ids = [Path(p).stem for p in sampled_tile_paths]

        return images, tile_ids, nb_tiles
