import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from ColorizationDataset import LABDataset
from model import Unet, Discriminator
import time
import psutil

def plot_training_curves(metrics):
    """Plot comprehensive training metrics."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(20, 10))
    
    # Plot Generator Losses
    plt.subplot(2, 2, 1)
    plt.plot(metrics['G_losses'], label='Total Loss', color='blue')
    plt.plot(metrics['G_GAN_losses'], label='GAN Loss', color='green')
    plt.plot(metrics['G_L1_losses'], label='L1 Loss', color='red')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Generator Losses')
    plt.legend()
    plt.grid(True)

    # Plot Discriminator Loss
    plt.subplot(2, 2, 2)
    plt.plot(metrics['D_losses'], label='Discriminator Loss', color='purple')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Discriminator Loss')
    plt.legend()
    plt.grid(True)

    # Plot Validation Loss
    plt.subplot(2, 2, 3)
    plt.plot(metrics['val_losses'], label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()
    plt.grid(True)

    # Plot Training Time
    plt.subplot(2, 2, 4)
    plt.plot(metrics['epoch_times'], label='Time per Epoch', color='brown')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title('Training Time per Epoch')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'training_metrics_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()

def initialize_data_loaders():
    """Initialize data loaders optimized for H100."""
    train_l_path = "../../preprocessed_data/training/L/"
    train_ab_path = "../../preprocessed_data/training/AB/"
    valid_l_path = "../../preprocessed_data/validation/L/"
    valid_ab_path = "../../preprocessed_data/validation/AB/"
    
    train_dataset = LABDataset(train_l_path, train_ab_path)
    valid_dataset = LABDataset(valid_l_path, valid_ab_path)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,  
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=64,  # Back to same as training
        shuffle=False,
        num_workers=32,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False
    )
    
    return train_loader, valid_loader

def initialize_model():
    """Initialize models optimized for H100."""
    generator = Unet(input_nc=1, output_nc=2, num_downs=7, ngf=64)
    discriminator = Discriminator(input_nc=3, ndf=64, n_layers=3)
    
    # Enable CUDA optimizations for RNNs
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    generator = generator.to(device, memory_format=torch.channels_last)
    discriminator = discriminator.to(device, memory_format=torch.channels_last)
    # Transfer to device with optimized memory format
    
    return generator, discriminator

def train_model(generator, discriminator, train_loader, valid_loader, optimizer_G, optimizer_D, num_epochs, scheduler=None):
    """Training loop optimized for H100 GPU with comprehensive metrics tracking."""
    scaler = GradScaler()
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()
    
    # Initialize metrics dictionary
    metrics = {
        'G_losses': [],
        'G_GAN_losses': [],
        'G_L1_losses': [],
        'D_losses': [],
        'val_losses': [],
        'epoch_times': []
    }
        

    
    for epoch in range(num_epochs):
        gpu_mem = torch.cuda.memory_allocated() / 1024**2
        ram_used = psutil.Process().memory_info().rss / 1024**2
        print(f"GPU Memory: {gpu_mem:.0f}MB, RAM Used: {ram_used:.0f}MB")
        epoch_start_time = time.time()
        generator.train()
        discriminator.train()
        
        epoch_G_loss = 0
        epoch_D_loss = 0
        epoch_G_GAN_loss = 0
        epoch_G_L1_loss = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for i, (L, AB) in enumerate(pbar):
                L = L.to(device, non_blocking=True, memory_format=torch.channels_last)
                AB = AB.to(device, non_blocking=True, memory_format=torch.channels_last)
                
                # Train discriminator
                optimizer_D.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    fake_AB = generator(L)
                    real_input = torch.cat((L, AB), dim=1)
                    fake_input = torch.cat((L, fake_AB.detach()), dim=1)
                    
                    pred_real = discriminator(real_input)
                    pred_fake = discriminator(fake_input)
                    
                    loss_D = (criterion_GAN(pred_real, torch.ones_like(pred_real)) + 
                             criterion_GAN(pred_fake, torch.zeros_like(pred_fake))) * 0.5
                
                scaler.scale(loss_D).backward()
                scaler.step(optimizer_D)
                
                # Train generator
                optimizer_G.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    fake_AB = generator(L)
                    fake_input = torch.cat((L, fake_AB), dim=1)
                    pred_fake = discriminator(fake_input)
                    
                    loss_G_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
                    loss_G_L1 = criterion_L1(fake_AB, AB) * 100
                    loss_G = loss_G_GAN + loss_G_L1
                
                scaler.scale(loss_G).backward()
                scaler.step(optimizer_G)
                scaler.update()
                
                # Accumulate losses
                epoch_G_loss += loss_G.item()
                epoch_D_loss += loss_D.item()
                epoch_G_GAN_loss += loss_G_GAN.item()
                epoch_G_L1_loss += loss_G_L1.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'G_loss': loss_G.item(),
                    'D_loss': loss_D.item(),
                    'L1_loss': loss_G_L1.item()
                })
                        
        # Calculate average losses for the epoch
        num_batches = len(train_loader)
        avg_G_loss = epoch_G_loss / num_batches
        avg_D_loss = epoch_D_loss / num_batches
        avg_G_GAN_loss = epoch_G_GAN_loss / num_batches
        avg_G_L1_loss = epoch_G_L1_loss / num_batches
        
        # Validation phase
        generator.eval()
        val_loss = 0
        with torch.no_grad(), autocast(dtype=torch.float16):
            for L, AB in valid_loader:
                L = L.to(device, non_blocking=True, memory_format=torch.channels_last)
                AB = AB.to(device, non_blocking=True, memory_format=torch.channels_last)
                fake_AB = generator(L)
                val_loss += criterion_L1(fake_AB, AB).item()
        
        val_loss /= len(valid_loader)
        epoch_time = time.time() - epoch_start_time
        
        # Store metrics
        metrics['G_losses'].append(avg_G_loss)
        metrics['G_GAN_losses'].append(avg_G_GAN_loss)
        metrics['G_L1_losses'].append(avg_G_L1_loss)
        metrics['D_losses'].append(avg_D_loss)
        metrics['val_losses'].append(val_loss)
        metrics['epoch_times'].append(epoch_time)
        
        # Print epoch summary
        print(f"\nEpoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.2f}s")
        print(f"G_loss: {avg_G_loss:.4f}, D_loss: {avg_D_loss:.4f}, Val_loss: {val_loss:.4f}")
        
        if scheduler:
            scheduler.step()
        
        # Save checkpoints with metrics
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'metrics': metrics,
                'val_loss': val_loss,
                'scaler_state_dict': scaler.state_dict(),
            }, f'checkpoint_epoch_{epoch+1}.pt')
            
            # Plot current metrics
            plot_training_curves(metrics)

    # Final metrics plot
    plot_training_curves(metrics)
    return metrics

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    # Training parameters optimized for H100
    learning_rate = 0.0001  # Slightly higher learning rate due to larger batch size
    num_epochs = 50
    device = torch.device("cuda")
    
    train_loader, valid_loader = initialize_data_loaders()
    generator, discriminator = initialize_model()
    
    # Use AdamW instead of Adam for better performance
    optimizer_G = optim.AdamW(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_D = optim.AdamW(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    
    # Use OneCycleLR for faster convergence
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer_G,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3
    )
    
    train_model(generator, discriminator, train_loader, valid_loader, 
                optimizer_G, optimizer_D, num_epochs, scheduler)