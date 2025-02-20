import random
import numpy as np
from PIL import Image

from histopathossl.training.dataset import SuperpixelMoCoDataset, SuperpixelMoCoDatasetFaster

# Fix the random seed to control for random.choice
random.seed(42)

# Use a deterministic transform or disable it altogether for testing.
transform = None  # or define a deterministic transform if needed

# Instantiate both datasets with the same mapping file and transform
dataset1 = SuperpixelMoCoDataset("/home/valentin/workspaces/histolung/data/interim/tiles_superpixels_with_overlap/superpixel_mapping_train.json", transform=transform)
dataset2 = SuperpixelMoCoDatasetFaster("/home/valentin/workspaces/histolung/data/interim/tiles_superpixels_with_overlap/superpixel_mapping_train.json", transform=transform)

# Choose an index to test (e.g., the first sample)
idx = 0

# Retrieve image pairs from both datasets
img1_a, img1_b = dataset1[idx]
img2_a, img2_b = dataset2[idx]

# Convert images to NumPy arrays for comparison
img1_a_array = np.array(img1_a)
img1_b_array = np.array(img1_b)
img2_a_array = np.array(img2_a)
img2_b_array = np.array(img2_b)

# Compare the arrays
assert np.array_equal(img1_a_array, img2_a_array), "First images differ!"
assert np.array_equal(img1_b_array, img2_b_array), "Second images differ!"

print("Both dataloaders returned the same outputs for index", idx)
