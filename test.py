import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
from model import Unet
import matplotlib.pyplot as plt
"""
This File will be used for manual inference of the model, 1 image at a time
"""

model_path = "generator_final.pth"
#model_path = "generator_epoch_30.pth"
# replace with the path of the image you want to run inference on
#input_image_path = "colorization/training_small/acorn/002.jpg"
input_image_path = "colorization/training_small/zebra/002.jpg"
#input_image_path = "colorization/validation_small/yorkshire_terrier/000.jpg"
input_image_path = "colorization/validation_small/black_swan/001.jpg"
output_image_path = "modeltest1.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = Unet(input_nc=1, output_nc=2, num_downs=7, ngf=64)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

def preprocess_image(image_path):
    """
    Preprocess image to match training preprocessing
    Returns 4D tensor [batch, channel, height, width]
    """
    # Read as RGB and resize to match training
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    image = cv2.resize(image, (256, 256))
    
    # Convert RGB to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    
    # Normalize L to [0, 1]
    L_tensor = torch.from_numpy(L).float() / 100.0
    L_tensor = L_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    
    return L_tensor, L

def chroma_hue_to_lab(chroma, hue):
    """Convert from Chroma/Hue to LAB color space"""
    # Chroma comes from sigmoid [0,1], scale to [0,128]
    chroma = np.clip(chroma * 128.0, 0, 128.0)
    
    # Hue comes from tanh [-1,1], scale to [-π, π]
    hue = hue * np.pi
    
    # Convert polar to cartesian
    a = chroma * np.cos(hue)
    b = chroma * np.sin(hue)
    
    # Center at 128 and clip to valid range
    a = np.clip(a + 128.0, 0, 255)
    b = np.clip(b + 128.0, 0, 255)
    
    return a, b

def postprocess_output(L_channel, chroma_hue_tensor):
    """
    Convert model output back to RGB image
    """
    # Split chroma and hue predictions
    chroma = chroma_hue_tensor[0].squeeze().detach().cpu().numpy()
    hue = chroma_hue_tensor[1].squeeze().detach().cpu().numpy()
    print(f"Chroma stats - min: {chroma.min():.3f}, max: {chroma.max():.3f}, mean: {chroma.mean():.3f}")
    print(f"Hue stats - min: {hue.min():.3f}, max: {hue.max():.3f}, mean: {hue.mean():.3f}")
    # Convert Chroma/Hue to LAB
    a, b = chroma_hue_to_lab(chroma, hue)
    
    # Create LAB image
    LAB = np.zeros((L_channel.shape[0], L_channel.shape[1], 3), dtype=np.uint8)
    LAB[:, :, 0] = L_channel
    LAB[:, :, 1] = a.astype(np.uint8)
    LAB[:, :, 2] = b.astype(np.uint8)
    
    # Convert to RGB
    RGB = cv2.cvtColor(LAB, cv2.COLOR_LAB2RGB)
    return RGB

with torch.no_grad():
    L_tensor, original_L = preprocess_image(input_image_path)
    L_tensor = L_tensor.to(device)
    
    # Forward pass - note the change here
    chroma_tensor, hue_tensor = model(L_tensor)  # Model returns tuple now
    
    # Postprocess output to get colorized image
    colorized_image = postprocess_output(original_L, (chroma_tensor, hue_tensor))

    grayscale_image = cv2.cvtColor(original_L[..., np.newaxis], cv2.COLOR_GRAY2RGB)
    
    # Combine original and colorized output side by side
    side_by_side_image = np.hstack((grayscale_image, colorized_image))
    
    # Save the side-by-side image
    cv2.imwrite(output_image_path, cv2.cvtColor(side_by_side_image, cv2.COLOR_RGB2BGR))
    print(f"Side-by-side image saved to {output_image_path}")