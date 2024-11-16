import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from model import Unet
from ColorizationDataset import LABDataset

# Training parameters
num_epochs = 50
batch_size = 64  # Increased from 16 to 64
learning_rate = 0.0008  # Adjusted to match increased batch size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_data_loaders():
    """Initialize the dataset and data loaders."""
    train_l_path = "preprocessed_data/training/L/"
    train_ab_path = "preprocessed_data/training/AB/"
    
    train_dataset = LABDataset(train_l_path, train_ab_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    return train_loader

def initialize_model():
    """Initialize the U-Net model and transfer it to the GPU if available."""
    model = Unet(input_nc=1, output_nc=2, num_downs=7, ngf=64)
    return model.to(device)

def train_model(model, train_loader, optimizer, scheduler, num_epochs):
    """Train the model with the given parameters."""
    # Initialize GradScaler for mixed precision
    scaler = GradScaler("cuda")
    
    # To track losses and time
    losses = []
    epoch_times = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        start_time = time.time()

        for L, AB in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", position=0, leave=True):
            # Move data to GPU if available
            L, AB = L.to(device), AB.to(device)
            optimizer.zero_grad()

            # Forward pass with mixed precision
            with autocast("cuda"):
                output = model(L)
                loss = criterion(output, AB)

            # Backward pass with scaled gradients
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        # Calculate average loss for the epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        losses.append(avg_epoch_loss)
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        # Log the loss and time taken for each epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}, Time: {epoch_time:.2f}s")

        # Step the scheduler at the end of each epoch
        scheduler.step()

        # Save model checkpoint periodically
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"unet_epoch_{epoch + 1}.pth")
            print(f"Model checkpoint saved at epoch {epoch + 1}")

    # Save the final model
    torch.save(model.state_dict(), "unet_colorization_final.pth")

    # Plot and save the loss curve and epoch time
    plot_training_curves(num_epochs, losses, epoch_times)

def plot_training_curves(num_epochs, losses, epoch_times):
    """Plot training loss and epoch time curves."""
    # Plot Loss Curve
    plt.figure()
    plt.plot(range(1, num_epochs + 1), losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig('training_loss_curve.png')
    plt.show()

    # Plot Time per Epoch
    plt.figure()
    plt.plot(range(1, num_epochs + 1), epoch_times, label='Time per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title('Epoch Time')
    plt.legend()
    plt.savefig('epoch_time.png')
    plt.show()

if __name__ == '__main__':
    # Initialize the data loader
    train_loader = initialize_data_loaders()

    # Initialize the model
    model = initialize_model()

    # Define Loss Function and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # Initialize Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Train the model
    train_model(model, train_loader, optimizer, scheduler, num_epochs)
