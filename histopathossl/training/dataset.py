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
            augmentation (callable, optional): augmentation to apply to each tile image.
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
