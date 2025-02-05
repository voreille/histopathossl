import json
import random

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
