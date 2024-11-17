import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
from model import Unet  
import matplotlib.pyplot as plt

model_path = "unet_colorization_final.pth"  
# path to the image, can either be grayscale or color
input_image_path = "colorization/training_small/monarch/163.jpg"
output_image_path = "modeltest.png" # Path to save the colorized output image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = Unet(input_nc=1, output_nc=2, num_downs=7, ngf=64)  # Adjust parameters to match your training model
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()


# Preprocessing function for input image
def preprocess_image(image_path):
    # Load image as grayscale
    image = Image.open(image_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to model's input size
        transforms.ToTensor(),         
        transforms.Normalize((0,), (1,))  # Normalize L to [0, 1]
    ])
    L_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    L_debug = L_tensor.squeeze().detach().cpu().numpy() * 255.0
    plt.imshow(L_debug, cmap="gray")
    plt.savefig("grayscale.png")

    return L_tensor

# convert LAB to RGB
def postprocess_output(L_tensor, AB_tensor):
    # denormalize L channel
    L = L_tensor.squeeze().detach().cpu().numpy() * 50.0 + 50.0
    # denormalize AB channels
    AB = AB_tensor.squeeze().detach().cpu().numpy() * 128.0
    # combine L and AB channels
    LAB = np.zeros((256, 256, 3), dtype=np.float32)
    LAB[..., 0] = L
    LAB[..., 1:] = AB.transpose(1, 2, 0)
    # convert LAB to RGB
    RGB = cv2.cvtColor(LAB.astype(np.uint8), cv2.COLOR_LAB2RGB)
    return RGB

with torch.no_grad():
    # Preprocess input image
    L_tensor = preprocess_image(input_image_path).to(device)
    
    # Forward pass through the model
    AB_tensor = model(L_tensor)
    
    # Postprocess output to get colorized image
    colorized_image = postprocess_output(L_tensor[0], AB_tensor[0])
    
    # Save or display the colorized image
    cv2.imwrite(output_image_path, cv2.cvtColor(colorized_image, cv2.COLOR_RGB2BGR))
    print(f"Colorized image saved to {output_image_path}")
