from pathlib import Path
import psutil
import os

import torch
import openslide
import numpy as np
import cv2
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image


class DummyDataset(Dataset):
    """
    A dataset that draws two random tiles from the same superpixel and applies MoCo transformations.

    Args:
        mapping_json (str): Path to the JSON file containing the mapping.
        transform (callable, optional): MoCo-style augmentation transform.
    """

    def __init__(self, transform=None, length=100, tile_size=224):
        # Load the mapping as a list of dictionaries

        self.transform = transform
        self.tile_size = tile_size
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Returns two different tiles belonging to the same superpixel.

        Returns:
            (Tensor, Tensor): Two augmented images.
        """

        # Sample two tiles (with replacement)

        # Load images
        image_1 = Image.fromarray(
            np.random.randint(
                0,
                high=255,
                size=(self.tile_size, self.tile_size, 3),
            ).astype(np.uint8)).convert("RGB")

        image_2 = Image.fromarray(
            np.random.randint(
                0,
                high=255,
                size=(self.tile_size, self.tile_size, 3),
            ).astype(np.uint8)).convert("RGB")
        # Apply MoCo transformations
        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)

        return image_1, image_2


class WSIDataset(Dataset):

    def __init__(self,
                 wsi_paths,
                 mask_paths,
                 tile_size=224,
                 target_magnification=10,
                 transform=None,
                 jitter=16,
                 num_tiles=2,
                 return_position=False,
                 return_wsi_id=False):
        """
        Args:
            wsi_paths (list of str or Path): List of WSI file paths.
            mask_paths (list of str or Path): List of binary mask file paths.
            tile_size (int): Size of each extracted tile.
            target_magnification (int): Desired magnification level (e.g., 10x, 20x).
            transform (callable, optional): Transformations to apply to the tiles.
            jitter (int): Maximum random offset (in pixels) to add to the tile center.
            num_tiles (int): Number of tiles to return per WSI.
            return_position (bool): Whether to return tile positions.
            return_wsi_id (bool): Whether to return the WSI ID.
        """
        self.wsi_paths = [Path(p) for p in wsi_paths]
        self.mask_paths = [Path(p) for p in mask_paths]
        self.tile_size = tile_size
        self.target_magnification = target_magnification
        self.transform = transform
        self.jitter = jitter
        self.num_tiles = num_tiles
        self.return_position = return_position
        self.return_wsi_id = return_wsi_id
        self.levels = []  # Stores optimal levels for each WSI

        print("Loading WSIs into memory...")
        self.slides = [
            openslide.OpenSlide(str(p)) for p in tqdm(self.wsi_paths)
        ]

        print(
            f"Determining best levels for {target_magnification}x magnification..."
        )
        self.compute_best_levels()

        print("Precomputing valid positions at mask resolution...")
        self.valid_positions, self.scale_factors = self.precompute_valid_positions(
        )

        self.estimate_memory_usage()

    def __len__(self):
        return len(self.wsi_paths)

    def compute_best_levels(self):
        """
        Determines the best OpenSlide level corresponding to `target_magnification`.
        Uses OpenSlide's `get_best_level_for_downsample()`.
        """
        for slide in self.slides:
            base_magnification = int(
                slide.properties.get('openslide.objective-power',
                                     40))  # Assume 40x if unknown
            downsample_factor = base_magnification / self.target_magnification
            best_level = slide.get_best_level_for_downsample(downsample_factor)
            self.levels.append(best_level)

    def precompute_valid_positions(self):
        """
        Precompute valid tile positions at mask resolution and store them.
        Returns:
            valid_positions: List of valid tile centers (at mask resolution).
            scale_factors: List of (scale_x, scale_y) for mapping to WSI resolution.
        """
        valid_positions = []
        scale_factors = []

        for slide, mask_path in zip(self.slides, self.mask_paths):
            # Load mask as binary image (0 or 1)
            mask = np.array(Image.open(mask_path).convert("L"))
            mask = (mask > 128).astype(np.uint8)  # Convert to binary mask

            # Get mask resolution and WSI level 0 resolution
            mask_h, mask_w = mask.shape
            wsi_w, wsi_h = slide.level_dimensions[0]

            # Compute scaling factors (from mask resolution to WSI level 0)
            scale_x = wsi_w / mask_w
            scale_y = wsi_h / mask_h
            scale_factors.append((scale_x, scale_y))

            # Get valid tile centers at mask resolution
            y_coords, x_coords = np.where(mask == 1)
            valid_positions.append(np.stack([x_coords, y_coords], axis=1))

        return valid_positions, scale_factors

    def get_valid_tile(self, slide_idx):
        """
        Selects a random valid tile center from the mask, scales it to WSI resolution,
        and adds random jitter.
        """
        if len(self.valid_positions[slide_idx]) == 0:
            raise ValueError(
                f"No valid positions found in mask for {self.wsi_paths[slide_idx]}."
            )

        # Select a random valid center at mask resolution
        x_mask, y_mask = self.valid_positions[slide_idx][np.random.randint(
            len(self.valid_positions[slide_idx]))]

        # Scale up to WSI level 0
        scale_x, scale_y = self.scale_factors[slide_idx]
        x = int(x_mask * scale_x)
        y = int(y_mask * scale_y)

        # Apply jitter (random shift) within a small range
        x += np.random.randint(-self.jitter, self.jitter)
        y += np.random.randint(-self.jitter, self.jitter)

        # Ensure tile is within WSI bounds
        slide = self.slides[slide_idx]
        level = self.levels[slide_idx]
        wsi_w, wsi_h = slide.level_dimensions[level]
        x = np.clip(x, 0, wsi_w - self.tile_size)
        y = np.clip(y, 0, wsi_h - self.tile_size)

        return x, y, level

    def estimate_memory_usage(self):
        """
        Estimate the memory required for storing all WSIs in RAM.
        """
        total_size = 0
        for slide in self.slides:
            base_magnification = int(
                slide.properties.get('openslide.objective-power',
                                     40))  # Assume 40x if unknown
            downsample_factor = base_magnification / self.target_magnification

            w, h = slide.level_dimensions[0]
            w /= downsample_factor
            h /= downsample_factor
            num_pixels = int(w) * int(h)
            total_size += num_pixels * 3  # Assuming RGB (3 bytes per pixel)

        total_size_gb = total_size / (1024**3)  # Convert to GB
        print(f"Estimated memory usage for all WSIs: {total_size_gb:.2f} GB")

        # Check actual memory usage
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info().rss / (1024**3)  # Convert to GB
        print(f"Current process memory usage: {mem_info:.2f} GB")

    def __getitem__(self, index):
        """
        Retrieves `num_tiles` random tiles from the WSI at `index`, using `read_region()`.
        Returns:
            tiles (list of Tensor): Extracted image tiles.
            wsi_id (str, optional): WSI filename (without extension).
            coords (list of tuples, optional): (x, y) coordinates of tile centers.
        """
        # wsi_id = self.wsi_paths[index].stem
        slide = self.slides[index]
        tiles, coords = [], []

        for _ in range(self.num_tiles):
            x, y, level = self.get_valid_tile(index)

            # Extract tile using OpenSlide read_region()
            tile = slide.read_region(
                (x, y), level, (self.tile_size, self.tile_size)).convert("RGB")
            # tile = Image.fromarray(
            #     np.random.rand(self.tile_size, self.tile_size,
            #                    3).astype(np.uint8))

            if self.transform:
                tile = self.transform(tile)

            tiles.append(tile)
            # coords.append((x, y))

        if self.num_tiles == 1:
            tiles = tiles[0]  # Return a single tile instead of a list

        # if self.return_wsi_id and self.return_position:
        #     return tiles, wsi_id, coords
        # elif self.return_wsi_id:
        #     return tiles, wsi_id
        # elif self.return_position:
        #     return tiles, coords
        # else:
        # return tiles
        return tiles
