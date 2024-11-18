import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
from model import Unet  
import matplotlib.pyplot as plt

model_path = "unet_epoch_10.pth"  
input_image_path = "colorization/training_small/turnstile/016.jpg"
output_image_path = "modeltest.png"  # Path to save the side-by-side output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = Unet(input_nc=1, output_nc=2, num_downs=7, ngf=64)  # Adjust parameters to match your training model
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Preprocessing function for input image
def preprocess_image(image_path):
    """
    Preprocess image to match training data normalization
    """
    image = Image.open(image_path).convert('L')
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0, 1] range
    ])
    L_tensor = transform(image).unsqueeze(0)
    return L_tensor

# Convert LAB to RGB
def postprocess_output(L_tensor, AB_tensor):
    """
    Convert model output back to RGB image
    """
    L = L_tensor.squeeze().detach().cpu().numpy() * 100.0
    AB = AB_tensor.squeeze().detach().cpu().numpy() * 128.0 + 128.0
    
    LAB = np.zeros((L.shape[0], L.shape[1], 3), dtype=np.float32)
    LAB[..., 0] = L
    LAB[..., 1:] = AB.transpose(1, 2, 0)
    
    LAB[..., 0] = np.clip(LAB[..., 0], 0, 100)
    LAB[..., 1:] = np.clip(LAB[..., 1:], -128, 127)
    
    LAB = LAB.astype(np.uint8)
    RGB = cv2.cvtColor(LAB, cv2.COLOR_LAB2RGB)
    return RGB

with torch.no_grad():
    # Preprocess input image
    L_tensor = preprocess_image(input_image_path).to(device)
    
    # Forward pass through the model
    AB_tensor = model(L_tensor)
    
    # Postprocess output to get colorized image
    colorized_image = postprocess_output(L_tensor[0], AB_tensor[0])
    
    # Prepare the L-channel for side-by-side display
    L_image = (L_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
    L_image = cv2.cvtColor(L_image, cv2.COLOR_GRAY2RGB)  # Convert to RGB for consistency
    
    # Combine L-channel and colorized output side by side
    side_by_side_image = np.hstack((L_image, colorized_image))
    
    # Save the side-by-side image
    cv2.imwrite(output_image_path, cv2.cvtColor(side_by_side_image, cv2.COLOR_RGB2BGR))
    print(f"Side-by-side image saved to {output_image_path}")
