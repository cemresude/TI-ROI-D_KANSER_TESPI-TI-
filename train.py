import torch
import torch.optim as optim
from tqdm import tqdm
import os

from model import ConvVAE, vae_loss
from data_loader import get_dataloaders
from config import Config
from utils import set_seed, save_checkpoint

def train_epoch(model, train_loader, optimizer, device, beta):
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    for images, _ in tqdm(train_loader, desc='Training'):
        images = images.to(device)
        
        optimizer.zero_grad()
        recon_images, mu, logvar = model(images)
        loss, recon_loss, kl_loss = vae_loss(recon_images, images, mu, logvar, beta)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
    
    avg_loss = total_loss / len(train_loader.dataset)
    avg_recon = total_recon_loss / len(train_loader.dataset)
    avg_kl = total_kl_loss / len(train_loader.dataset)
    
    return avg_loss, avg_recon, avg_kl

def validate(model, val_loader, device, beta):
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    with torch.no_grad():
        for images, _ in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            recon_images, mu, logvar = model(images)
            loss, recon_loss, kl_loss = vae_loss(recon_images, images, mu, logvar, beta)
            
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
    
    # Sadece BENIGN verilerle veri yükleyicileri oluştur
    print("Loading BENIGN data only for training...")
    train_loader, val_loader, train_size, val_size = get_dataloaders(
        Config.DATA_DIR, 
        Config.BATCH_SIZE, 
        Config.SPLIT_RATIO,
        only_benign=True  # Sadece benign veriler
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Model, optimizer
    model = ConvVAE(latent_dim=Config.LATENT_DIM).to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5
    )
    
    # Checkpoint dizini oluştur
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(Config.NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{Config.NUM_EPOCHS}')
        
        train_loss, train_recon, train_kl = train_epoch(
            model, train_loader, optimizer, Config.DEVICE, Config.BETA
        )
        val_loss, val_recon, val_kl = validate(
            model, val_loader, Config.DEVICE, Config.BETA
        )
        
        print(f'Train - Loss: {train_loss:.4f} | Recon: {train_recon:.4f} | KL: {train_kl:.4f}')
        print(f'Val   - Loss: {val_loss:.4f} | Recon: {val_recon:.4f} | KL: {val_kl:.4f}')
        
        scheduler.step(val_loss)
        
        # En iyi modeli kaydet
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
