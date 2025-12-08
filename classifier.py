import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
from optuna import optim

class ThyroidResNetClassifier(nn.Module):
    """
    ResNet18/50 transfer learning ile tiroid kanseri sınıflandırıcı
    """
    def __init__(self, num_classes=2, backbone='resnet18', pretrained=True):
        super(ThyroidResNetClassifier, self).__init__()
        
        # Backbone seç
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = 512
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Son FC layer'ı değiştir
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Grad-CAM için son conv layer'ı sakla
        self.feature_layers = None
        self.gradients = None
        
    def forward(self, x):
        # ResNet forward pass
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        # Son conv layer'ı kaydet (Grad-CAM için)
        self.feature_layers = x
        
        # Hook gradient
        if x.requires_grad:
            x.register_hook(self.save_gradient)
        
        # Classification head
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)
        
        return x
    
    def save_gradient(self, grad):
        """Gradient hook for Grad-CAM"""
        self.gradients = grad
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self):
        return self.feature_layers

class TemperatureScaling(nn.Module):
    """
    Temperature scaling for probability calibration
    https://arxiv.org/abs/1706.04599
    """
    def __init__(self, model, temperature=1.5):
        super(TemperatureScaling, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * temperature)
    
    def forward(self, x):
        logits = self.model(x)
        return logits / self.temperature
    
    def set_temperature(self, valid_loader, device):
        """
        Validation set ile optimal temperature bul
        """
        nll_criterion = nn.CrossEntropyLoss()
        
        logits_list = []
        labels_list = []
        
        self.model.eval()
        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(device)
                logits = self.model(images)
                logits_list.append(logits)
                labels_list.append(labels)
        
        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)
        
        # Optimize temperature
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(logits / self.temperature, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval)
        
        return self.temperature.item()

class GradCAM:
    """
    Grad-CAM implementasyonu
    https://arxiv.org/abs/1610.02391
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()
    
    def generate_cam(self, input_image, target_class=None):
        """
        Grad-CAM ısı haritası oluştur
        
        Args:
            input_image: [1, 3, H, W] tensor
            target_class: Hedef sınıf (None ise predicted class)
        
        Returns:
            cam: [H, W] numpy array (0-1 normalized)
        """
        # Forward pass
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        
        # Gradients ve activations al
        gradients = self.model.get_activations_gradient()
        activations = self.model.get_activations()
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        
        # Weighted combination
        cam = torch.sum(weights * activations, dim=1).squeeze()
        
        # ReLU
        cam = torch.clamp(cam, min=0)
        
        # Normalize
        cam = cam.cpu().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Resize to input size
        cam = cv2.resize(cam, (input_image.shape[3], input_image.shape[2]))
        
        return cam, target_class
    
    def generate_heatmap_overlay(self, original_image, cam, alpha=0.5):
        """
        Orijinal görüntü üzerine ısı haritası bindirme
        
        Args:
            original_image: [3, H, W] tensor veya [H, W, 3] numpy
            cam: [H, W] numpy array
            alpha: Overlay transparency
        
        Returns:
            overlay: [H, W, 3] numpy array
        """
        # Tensor ise numpy'a çevir
        if isinstance(original_image, torch.Tensor):
            original_image = original_image.cpu().detach().permute(1, 2, 0).numpy()
        
        # Normalize et [0, 1]
        if original_image.max() > 1.0:
            original_image = original_image / 255.0
        
        # Heatmap oluştur (jet colormap)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        
        # Overlay
        overlay = original_image * (1 - alpha) + heatmap * alpha
        overlay = np.clip(overlay, 0, 1)
        
        return overlay

def compute_classification_metrics(model, data_loader, device):
    """
    Sınıflandırıcı metrikleri hesapla
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Malignant probability
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)
