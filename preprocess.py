import os
import cv2
import numpy as np
import torch
from torchvision import datasets
from tqdm import tqdm

# Create directories for L and AB tensors if they do not exist
def create_output_directories(output_path):
    l_path = os.path.join(output_path, 'L')
    ab_path = os.path.join(output_path, 'AB')
    os.makedirs(l_path, exist_ok=True)
    os.makedirs(ab_path, exist_ok=True)
    return l_path, ab_path

def preprocess_and_save(data_path, l_output_path, ab_output_path):
    dataset = datasets.ImageFolder(data_path)

    # Iterate over all images in the dataset
    for idx, (rgb, _) in tqdm(enumerate(dataset), total=len(dataset), desc=f"Processing {data_path}"):
        # Convert the PIL image to a NumPy array
        rgb_np = np.array(rgb)

        # Convert RGB to LAB using OpenCV
        lab = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)  # Note OpenCV expects RGB here because PIL uses RGB
        L, A, B = cv2.split(lab)

        # Convert L, A, and B channels to PyTorch tensors
        L_tensor = torch.from_numpy(L).unsqueeze(0).float()  # Shape: (1, height, width)
        AB_tensor = torch.from_numpy(np.stack((A, B), axis=0)).float()  # Shape: (2, height, width)

        # Normalize the tensors
        L_tensor = L_tensor / 255.0  # Normalize L to range [0, 1]
        AB_tensor = (AB_tensor - 128.0) / 128.0  # Normalize AB to roughly [-1, 1]

        # Define paths for saving tensors
        L_output_path = os.path.join(l_output_path, f"L_{idx}.pt")
        AB_output_path = os.path.join(ab_output_path, f"AB_{idx}.pt")

        # Save L and AB tensors separately
        torch.save(L_tensor, L_output_path)
        torch.save(AB_tensor, AB_output_path)

if __name__ == "__main__":
    train_path = "colorization/training_small/"  
    valid_path = "colorization/validation_small/"  
    test_path = "colorization/test_small/"  

    train_l_path, train_ab_path = create_output_directories("preprocessed_data/training/")
    valid_l_path, valid_ab_path = create_output_directories("preprocessed_data/validation/")
    test_l_path, test_ab_path = create_output_directories("preprocessed_data/test/")
    
    # Preprocess training, validation, and test sets
    preprocess_and_save(train_path, train_l_path, train_ab_path)
    preprocess_and_save(valid_path, valid_l_path, valid_ab_path)
    preprocess_and_save(test_path, test_l_path, test_ab_path)


