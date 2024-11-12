import torch.utils.data as data
from PIL import Image
import os
import numpy as np

def make_dataset(dir, max_dataset_size=float("inf")):
    images = []

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            images.append(path)

    images = images[:min(max_dataset_size, len(images))]

    # Convert images to grayscale/color pairs in LAB colorspace
    lab_pairs = []
    for img_path in images:
        image = Image.open(img_path).convert('RGB') 
        image_lab = image.convert('LAB')  

        image_lab_np = np.array(image_lab)
        L_channel = image_lab_np[:, :, 0]  # Grayscale (L channel)
        AB_channels = image_lab_np[:, :, 1:]  # Color (A and B channels)

        lab_pairs.append((L_channel, AB_channels))

    return lab_pairs
