import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from model import Unet
from ColorizationDataset import LABDataset

# Training parameters
num_epochs = 50
batch_size = 64
learning_rate = 0.0008 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_data_loaders():
    """Initialize the dataset and data loaders."""
    train_l_path = "preprocessed_data/training/L/"
    train_ab_path = "preprocessed_data/training/AB/"
    valid_l_path = "preprocessed_data/validation/L/"
    valid_ab_path = "preprocessed_data/validation/AB/"
    
    train_dataset = LABDataset(train_l_path, train_ab_path)
    valid_dataset = LABDataset(valid_l_path, valid_ab_path)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=7, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=7, pin_memory=True)
    
    return train_loader, valid_loader

def initialize_model():
    """Initialize the U-Net model and transfer it to the GPU if available."""
    model = Unet(input_nc=1, output_nc=2, num_downs=7, ngf=64)
    return model.to(device)

def train_model(model, train_loader, valid_loader, optimizer, num_epochs, scheduler=None):
    """Train the model with the given parameters."""
    # Initialize GradScaler for mixed precision
    scaler = GradScaler("cuda")
    
    train_losses = []
    val_losses = []
    epoch_times = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        start_time = time.time()

        for L, AB in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", position=0, leave=True):
            L, AB = L.to(device), AB.to(device)
            optimizer.zero_grad()

            # Forward pass with mixed precision
            with autocast(device_type="cuda"):
                output = model(L)
                loss = criterion(output, AB)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        # Calculate average training loss
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Validation loop
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for L, AB in valid_loader:
                L, AB = L.to(device), AB.to(device)
                output = model(L)
                loss = criterion(output, AB)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(valid_loader)
        
        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        # Log the metrics
        print(f"Epoch [{epoch+1}/{num_epochs}], Time: {epoch_time:.2f}s")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        #scheduler.step()

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"unet_epoch_{epoch + 1}.pth")
            print(f"Model checkpoint saved at epoch {epoch + 1}")

    torch.save(model.state_dict(), "unet_colorization_final.pth")

    plot_training_curves(num_epochs, train_losses, val_losses, epoch_times)

def plot_training_curves(num_epochs, train_losses, val_losses, epoch_times):
    """Plot training loss and epoch time curves."""
    # Plot Losses
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot Time per Epoch
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), epoch_times, label='Time per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title('Epoch Time')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

if __name__ == '__main__':
    train_loader, valid_loader = initialize_data_loaders()
    model = initialize_model()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)
    
    train_model(model, train_loader, valid_loader, optimizer, num_epochs)