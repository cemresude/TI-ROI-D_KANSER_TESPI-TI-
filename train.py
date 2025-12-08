import torch
import torch.optim as optim
from tqdm import tqdm
import os

from model import ConvVAE, vae_loss
from data_loader import get_dataloaders
from config import Config
from utils import set_seed, save_checkpoint

def get_beta_annealing(epoch, max_epochs, start_beta=0.0, end_beta=1.0, warmup_epochs=10):
    """
    Beta annealing: Linearly increase beta from start_beta to end_beta
    """
    if epoch < warmup_epochs:
        return start_beta + (end_beta - start_beta) * (epoch / warmup_epochs)
    return end_beta

def train_epoch(model, train_loader, optimizer, device, beta, use_ssim=True):
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    for images, _ in tqdm(train_loader, desc='Training'):
        images = images.to(device)
        
        optimizer.zero_grad()
        recon_images, mu, logvar = model(images)
        loss, recon_loss, kl_loss = vae_loss(recon_images, images, mu, logvar, beta, use_ssim=use_ssim)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
    
    avg_loss = total_loss / len(train_loader.dataset)
    avg_recon = total_recon_loss / len(train_loader.dataset)
    avg_kl = total_kl_loss / len(train_loader.dataset)
    
    return avg_loss, avg_recon, avg_kl

def validate(model, val_loader, device, beta, use_ssim=True):
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    with torch.no_grad():
        for images, _ in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            recon_images, mu, logvar = model(images)
            loss, recon_loss, kl_loss = vae_loss(recon_images, images, mu, logvar, beta, use_ssim=use_ssim)
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
    
    avg_loss = total_loss / len(val_loader.dataset)
    avg_recon = total_recon_loss / len(val_loader.dataset)
    avg_kl = total_kl_loss / len(val_loader.dataset)
    
    return avg_loss, avg_recon, avg_kl

def main():
    set_seed(Config.SEED)
    
    print(f"Device: {Config.DEVICE}")
    
    print("Loading BENIGN data only for training...")
    train_loader, val_loader, train_size, val_size = get_dataloaders(
        Config.DATA_DIR, 
        Config.BATCH_SIZE, 
        Config.SPLIT_RATIO,
        only_benign=True,
        use_weighted_sampler=False  # Benign only, no need for balancing
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    model = ConvVAE(latent_dim=Config.LATENT_DIM).to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5
    )
    
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\nTraining with Beta Annealing (start={Config.BETA_START}, end={Config.BETA_END})")
    print(f"Using SSIM loss: {Config.USE_SSIM}\n")
    
    for epoch in range(Config.NUM_EPOCHS):
        # Beta annealing
        current_beta = get_beta_annealing(
            epoch, Config.NUM_EPOCHS, 
            start_beta=Config.BETA_START, 
            end_beta=Config.BETA_END,
            warmup_epochs=Config.BETA_WARMUP_EPOCHS
        )
        
        print(f'\nEpoch {epoch+1}/{Config.NUM_EPOCHS} | Beta: {current_beta:.4f}')
        
        train_loss, train_recon, train_kl = train_epoch(
            model, train_loader, optimizer, Config.DEVICE, current_beta, use_ssim=Config.USE_SSIM
        )
        val_loss, val_recon, val_kl = validate(
            model, val_loader, Config.DEVICE, current_beta, use_ssim=Config.USE_SSIM
        )
        
        print(f'Train - Loss: {train_loss:.4f} | Recon: {train_recon:.4f} | KL: {train_kl:.4f}')
        print(f'Val   - Loss: {val_loss:.4f} | Recon: {val_recon:.4f} | KL: {val_kl:.4f}')
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss - Config.EARLY_STOPPING_DELTA:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch, val_loss, 
                Config.MODEL_SAVE_PATH, is_vae=True
            )
            print(f'Model saved! Best Val Loss: {best_val_loss:.4f}')
        else:
            patience_counter += 1
            print(f'No improvement. Patience counter: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}')
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break
    print("\nTraining completed!")

if __name__ == '__main__':
    main()
