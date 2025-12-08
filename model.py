import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim

class ThyroidCancerCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(ThyroidCancerCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Conv block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Conv block 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class ConvVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(ConvVAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder - 224x224 için güncellenmiş
        self.encoder = nn.Sequential(
            # 224x224x3 -> 112x112x32
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # 112x112x32 -> 56x56x64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # 56x56x64 -> 28x28x128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # 28x28x128 -> 14x14x256
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            # 14x14x256 -> 7x7x512
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        
        # 7x7x512 = 25088 (doğru!)
        self.fc_mu = nn.Linear(512 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(512 * 7 * 7, latent_dim)
        
        # Decoder input
        self.decoder_input = nn.Linear(latent_dim, 512 * 7 * 7)
        
        # Decoder - 224x224 için güncellenmiş
        self.decoder = nn.Sequential(
            # 7x7x512 -> 14x14x256
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            # 14x14x256 -> 28x28x128
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # 28x28x128 -> 56x56x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # 56x56x64 -> 112x112x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # 112x112x32 -> 224x224x3
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        """Encoder: görüntüden latent parametrelere"""
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # [batch, 512*7*7] = [batch, 25088]
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decoder: latent space'den görüntüye"""
        x = self.decoder_input(z)
        x = x.view(x.size(0), 512, 7, 7)  # [batch, 512, 7, 7]
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        """Forward pass"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def vae_loss(recon_x, x, mu, logvar, beta=1.0, use_ssim=True, use_mae=True, 
             ssim_weight=0.4, mae_weight=0.3, mse_weight=0.3):
    """
    İyileştirilmiş VAE Loss: MAE + SSIM + MSE + beta * KL
    """
    batch_size = x.size(0)
    
    # Reconstruction loss components
    recon_loss = 0.0
    
    # 1. MSE Loss
    if mse_weight > 0:
        mse_loss = F.mse_loss(recon_x, x, reduction='none')
        mse_loss = mse_loss.view(batch_size, -1).mean(dim=1).mean()
        recon_loss += mse_weight * mse_loss
    
    # 2. MAE Loss (L1)
    if use_mae and mae_weight > 0:
        mae_loss = F.l1_loss(recon_x, x, reduction='none')
        mae_loss = mae_loss.view(batch_size, -1).mean(dim=1).mean()
        recon_loss += mae_weight * mae_loss
    
    # 3. SSIM Loss
    if use_ssim and ssim_weight > 0:
        try:
            ssim_val = ssim(recon_x, x, data_range=1.0, size_average=True)
            ssim_loss = 1.0 - ssim_val
            recon_loss += ssim_weight * ssim_loss
        except Exception as e:
            print(f"SSIM failed: {e}")
    
    # KL Divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    assert total_loss.dim() == 0, f"Loss not scalar: {total_loss.shape}"
    
    return total_loss, recon_loss, kl_loss
