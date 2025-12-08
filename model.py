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
        
        # Encoder
        self.encoder = nn.Sequential(
            # 128x128x3 -> 64x64x32
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # 64x64x32 -> 32x32x64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # 32x32x64 -> 16x16x128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # 16x16x128 -> 8x8x256
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            # 8x8x256 -> 4x4x512
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        
        # Latent space (mean ve variance)
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)
        
        # Decoder input
        self.decoder_input = nn.Linear(latent_dim, 512 * 4 * 4)
        
        # Decoder
        self.decoder = nn.Sequential(
            # 4x4x512 -> 8x8x256
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            # 8x8x256 -> 16x16x128
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # 16x16x128 -> 32x32x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # 32x32x64 -> 64x64x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # 64x64x32 -> 128x128x3
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # 0-1 aralığında çıktı
        )
        
    def encode(self, x):
        """Encoder: görüntüden latent parametrelere"""
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + sigma * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decoder: latent space'den görüntüye"""
        x = self.decoder_input(z)
        x = x.view(x.size(0), 512, 4, 4)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        """Forward pass"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def vae_loss(recon_x, x, mu, logvar, beta=1.0, use_ssim=True, ssim_weight=0.5):
    """
    VAE Loss: (MSE + SSIM) + beta * KL Divergence
    Returns SCALAR losses averaged over the batch
    """
    batch_size = x.size(0)
    
    # Reconstruction loss
    if use_ssim:
        # MSE Loss (per sample, then mean)
        mse_loss = F.mse_loss(recon_x, x, reduction='none')
        mse_loss = mse_loss.view(batch_size, -1).mean(dim=1)  # [batch_size]
        mse_loss = mse_loss.mean()  # scalar
        
        # SSIM Loss
        try:
            ssim_val = ssim(recon_x, x, data_range=1.0, size_average=True)  # scalar
            ssim_loss = 1.0 - ssim_val  # scalar
        except Exception as e:
            print(f"SSIM computation failed: {e}. Falling back to MSE only.")
            ssim_loss = 0.0
        
        # Combined reconstruction loss (scalar)
        recon_loss = (1 - ssim_weight) * mse_loss + ssim_weight * ssim_loss
    else:
        # MSE only (scalar)
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    
    # KL Divergence (per sample, then mean)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # [batch_size]
    kl_loss = kl_loss.mean()  # scalar
    
    # Total loss (scalar)
    total_loss = recon_loss + beta * kl_loss
    
    # Ensure all are scalars
    assert total_loss.dim() == 0, f"total_loss is not scalar: shape={total_loss.shape}"
    assert recon_loss.dim() == 0, f"recon_loss is not scalar: shape={recon_loss.shape}"
    assert kl_loss.dim() == 0, f"kl_loss is not scalar: shape={kl_loss.shape}"
    
    return total_loss, recon_loss, kl_loss
