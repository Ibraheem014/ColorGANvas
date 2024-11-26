import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from model import Unet, Discriminator
from ColorizationDataset import LABDataset
from ChromaHueKLLoss import CromaHueKLLoss

"""
This file trains the model
"""

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

    # Initialize the model and the discriminator
    generator = Unet(input_nc=1, output_nc=2, num_downs=7, ngf=64)
    discriminator = Discriminator(input_nc=8, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d)

    # Transfer models to device
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    return generator, discriminator

def train_model(generator, discriminator, train_loader, valid_loader, optimizer_G, optimizer_D, num_epochs, scheduler=None):
    """Train the model with the given parameters."""
    # Initialize GradScaler for mixed precision
    scaler = GradScaler("cuda")

    # Initialize loss functions
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()
    chroma_hue_loss = CromaHueKLLoss(lambda_hue=5.0)
    
    # Initialize lists to store metrics
    train_G_losses = []
    train_D_losses = []
    val_losses = []
    epoch_times = []

    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        epoch_G_loss = 0
        epoch_D_loss = 0
        start_time = time.time()
        

        for L, AB in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", position=0, leave=True):
            L, AB = L.to(device), AB.to(device) #real AB input tensor size flow 4 
            optimizer_D.zero_grad()

            # Forward pass with mixed precision
            with autocast(device_type="cuda"):
                fake_chroma, fake_hue = generator(L)
                fake_AB = torch.cat((fake_chroma, fake_hue), dim=1)

                L_expanded = L.expand(-1, 4, -1, -1)
                #print("L shape:", L.shape)                   # Expected: (batch_size, 1, height, width)
                #print("fake_chroma shape:", fake_chroma.shape)  # Expected: (batch_size, height, width)
                #print("fake_hue shape:", fake_hue.shape)        # Expected: (batch_size, height, width)
                #print("fake_AB shape:", fake_AB.shape)          # Expected: (batch_size, 2, height, width)
                #print("L_expanded shape:", L_expanded.shape)    # Expected: (batch_size, 2, height, width)
                #print("AB shape:", AB.shape) #Expected: (batch_size, 4, height, width) currently 2®

                # Create real and fake input pairs for the discriminator
                chroma = AB[:, 0:1, :, :]
                hue = AB[:, 1:2, :, :]
                AB_expand = torch.cat((chroma, chroma, hue, hue), dim=1)
                real_input = torch.cat((L_expanded, AB_expand), dim=1)   # Real pair (L and real AB)
                fake_input = torch.cat((L_expanded, fake_AB.detach()), dim=1)  # Fake pair (L and fake AB)



                #print("real_input shape:", real_input.shape)    # Expected: (batch_size, 4, height, width)
                #print("fake_input shape:", fake_input.shape)    # Expected: (batch_size, 4, height, width)


                # Discriminator output for real images
                pred_real = discriminator(real_input)
                # Discriminator output for fake images
                pred_fake = discriminator(fake_input)

                # Labels for real and fake images
                real_labels = torch.ones_like(pred_real)
                fake_labels = torch.zeros_like(pred_fake)

                # Compute discriminator losses
                loss_D_real = criterion_GAN(pred_real, real_labels)
                loss_D_fake = criterion_GAN(pred_fake, fake_labels)
                
                loss_D = (loss_D_real + loss_D_fake) * 0.5

            # Backpropagate discriminator loss
            scaler.scale(loss_D).backward()
            scaler.step(optimizer_D)
            scaler.update()

            optimizer_G.zero_grad()

            # Generate fake images again for generator update
            with autocast(device_type="cuda"):
                fake_chroma, fake_hue = generator(L)
                fake_AB = torch.cat((fake_chroma, fake_hue), dim=1)
                L_expanded = L.expand(-1, 4, -1, -1)
                
                fake_input = torch.cat((L_expanded, fake_AB.detach()), dim=1)
                pred_fake = discriminator(fake_input)

                # Generator adversarial loss (we want the discriminator to classify the fake images as real)
                loss_G_GAN = criterion_GAN(pred_fake, real_labels)

                # Generator reconstruction loss
                ##loss_G_L1 = criterion_L1(fake_AB, AB) * 100  # Adjust the weight as needed

                # Separate Chroma and Hue components (ensure your model outputs these separately)
                gt_chroma, gt_hue = AB[:, 0, :, :].unsqueeze(1), AB[:, 1, :, :].unsqueeze(1)
                #fake_chroma = fake_chroma.unsqueeze(1)
                #fake_hue = fake_hue.unsqueeze(1)

                #print("gt_chroma shape:", gt_chroma.shape)
                #print("fake_chroma shape:", fake_chroma.shape)
                #print("gt_hue shape:", gt_hue.shape)
                #print("fake_hue shape:", fake_hue.shape)


                # Compute the Chroma-Hue KL loss
                chroma_values = gt_chroma  # Chroma values for weighting
                loss_G_KL = chroma_hue_loss(gt_chroma, fake_chroma, gt_hue, fake_hue, chroma_values)


                # Total generator loss
                loss_G = loss_G_GAN + loss_G_KL
            
            # Backpropagate generator loss
            scaler.scale(loss_G).backward()
            scaler.step(optimizer_G)
            scaler.update()

            # Accumulate losses
            epoch_G_loss += loss_G.item()
            epoch_D_loss += loss_D.item()

        # Calculate average losses
        avg_G_loss = epoch_G_loss / len(train_loader)
        avg_D_loss = epoch_D_loss / len(train_loader)
        
        # Validation loop
        generator.eval()
        val_loss = 0
        
        with torch.no_grad():
            for L, AB in valid_loader:
                L, AB = L.to(device), AB.to(device)
                fake_chroma, fake_hue = generator(L)
                gt_chroma, gt_hue = AB[:, 0, :, :], AB[:, 1, :, :]

                #output = generator(L)
                ##loss = criterion_L1(output, AB)
                ##val_loss += loss.item()
                # Separate Chroma and Hue
                # Ensure tensors have a channel dimension
                if gt_chroma.dim() == 3:  # If shape is [batch_size, height, width]
                    gt_chroma = gt_chroma.unsqueeze(1)  # Shape: [batch_size, 1, height, width]
                if gt_hue.dim() == 3:
                    gt_hue = gt_hue.unsqueeze(1)
                if fake_chroma.dim() == 3:
                    fake_chroma = fake_chroma.unsqueeze(1)
                if fake_hue.dim() == 3:
                    fake_hue = fake_hue.unsqueeze(1)

                chroma_values = gt_chroma
                val_loss += chroma_hue_loss(gt_chroma, fake_chroma, gt_hue, fake_hue, chroma_values).item()
                
            avg_val_loss = val_loss / len(valid_loader)
        
        # Store metrics
        train_G_losses.append(avg_G_loss)
        train_D_losses.append(avg_D_loss)
        val_losses.append(avg_val_loss)
        
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        # Log the metrics
        print(f"Epoch [{epoch+1}/{num_epochs}], Time: {epoch_time:.2f}s")
        print(f"Generator Loss: {avg_G_loss:.4f}, Discriminator Loss: {avg_D_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        #scheduler.step()

        # Save checkpoints
        if (epoch + 1) % 5 == 0:
            torch.save(generator.state_dict(), f"generator_epoch_{epoch + 1}.pth")
            torch.save(discriminator.state_dict(), f"discriminator_epoch_{epoch + 1}.pth")
            print(f"Model checkpoints saved at epoch {epoch + 1}")

    # Save final models
    torch.save(generator.state_dict(), "generator_final.pth")
    torch.save(discriminator.state_dict(), "discriminator_final.pth")

    # Plot training curves
    plot_training_curves(num_epochs, train_G_losses, train_D_losses, val_losses, epoch_times)

def plot_training_curves(num_epochs, train_G_losses, train_D_losses, val_losses, epoch_times):
    """Plot training loss and epoch time curves."""
    epochs = range(1, num_epochs + 1)

    # Plot Generator and Discriminator Losses
    plt.figure(figsize=(15, 5))

    # Generator Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_G_losses, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator Loss')
    plt.legend()

    # Discriminator Loss
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_D_losses, label='Discriminator Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Discriminator Loss')
    plt.legend()

    # Validation Loss
    plt.subplot(1, 3, 3)
    plt.plot(epochs, val_losses, label='Validation Loss', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()
    
    # Plot Time per Epoch
    plt.figure()
    plt.plot(epochs, epoch_times, label='Time per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title('Epoch Time')
    plt.legend()
    plt.savefig('epoch_times.png')
    plt.show()

if __name__ == '__main__':
    train_loader, valid_loader = initialize_data_loaders()
    generator, discriminator = initialize_model()
    criterion = nn.MSELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)
    
    train_model(generator, discriminator, train_loader, valid_loader, optimizer_G, optimizer_D, num_epochs)