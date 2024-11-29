import os
import cv2
import numpy as np
import torch
from torchvision import datasets
from tqdm import tqdm
import matplotlib.pyplot as plt

def verify_single_image(rgb_image, save_path=None):
    # Convert to LAB
    lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    
    # Split channels
    L, A, B = cv2.split(lab)
    
    # Create tensors with correct normalization
    L_tensor = torch.from_numpy(L).unsqueeze(0).float() / 100.0  # Normalize L to [0, 1]
    
    # Convert to Chroma/Hue
    A_shifted = A - 128.0
    B_shifted = B - 128.0
    chroma = np.sqrt(A_shifted**2 + B_shifted**2) / 128.0
    hue = (np.arctan2(B_shifted, A_shifted) + np.pi) / (2 * np.pi)
    CH_tensor = torch.from_numpy(np.stack((chroma, hue), axis=0)).float()
    
    # Convert back for verification
    L_recover = (L_tensor.squeeze().numpy() * 100.0).astype(np.uint8)
    chroma = CH_tensor[0].numpy() * 128.0
    hue = CH_tensor[1].numpy() * 2 * np.pi - np.pi
    
    A_recover = chroma * np.cos(hue) + 128.0
    B_recover = chroma * np.sin(hue) + 128.0
    
    lab_reconstructed = cv2.merge([L_recover, A_recover.astype(np.uint8), B_recover.astype(np.uint8)])
    rgb_reconstructed = cv2.cvtColor(lab_reconstructed, cv2.COLOR_LAB2RGB)
    
    if save_path:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(rgb_image)
        plt.title('Original')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(rgb_reconstructed)
        plt.title('Reconstructed')
        plt.axis('off')
        
        plt.savefig(save_path)
        plt.close()
    
    return rgb_reconstructed

def preprocess_and_save(data_path, l_output_path, ch_output_path):
    dataset = datasets.ImageFolder(data_path)
    os.makedirs("verification", exist_ok=True)

    for idx, (rgb, _) in tqdm(enumerate(dataset), total=len(dataset)):
        rgb_np = np.array(rgb)
        rgb_np = cv2.resize(rgb_np, (256, 256))
        
        if idx < 5:
            verify_single_image(rgb_np, f"verification/verify_{idx}.png")
        
        lab = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)
        L, A, B = cv2.split(lab)
        
        L_tensor = torch.from_numpy(L).unsqueeze(0).float() / 100.0

        A_shifted = A - 128.0
        B_shifted = B - 128.0
        chroma = np.sqrt(A_shifted**2 + B_shifted**2) / 128.0
        hue = (np.arctan2(B_shifted, A_shifted) + np.pi) / (2 * np.pi)
        
        CH_tensor = torch.from_numpy(np.stack((chroma, hue), axis=0)).float()

        torch.save(L_tensor, os.path.join(l_output_path, f"L_{idx}.pt"))
        torch.save(CH_tensor, os.path.join(ch_output_path, f"CH_{idx}.pt"))

if __name__ == "__main__":
    train_path = "colorization/training_small/"
    train_l_path = "preprocessed_data/training/L/"
    train_ch_path = "preprocessed_data/training/CH/"
    os.makedirs(train_l_path, exist_ok=True)
    os.makedirs(train_ch_path, exist_ok=True)

    valid_path = "colorization/validation_small/"
    valid_l_path = "preprocessed_data/validation/L/"
    valid_ch_path = "preprocessed_data/validation/CH/"
    os.makedirs(valid_l_path, exist_ok=True)
    os.makedirs(valid_ch_path, exist_ok=True)
    
    preprocess_and_save(train_path, train_l_path, train_ch_path)
    preprocess_and_save(valid_path, valid_l_path, valid_ch_path)