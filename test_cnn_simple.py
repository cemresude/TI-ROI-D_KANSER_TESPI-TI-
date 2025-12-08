# Hızlı CNN test scripti
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load models
from classifier import ThyroidResNetClassifier
from data_loader import get_test_loader
from config import Config

test_loader = get_test_loader(Config.DATA_DIR, Config.BATCH_SIZE)

model = ThyroidResNetClassifier(num_classes=2, backbone='resnet18', pretrained=False).to(Config.DEVICE)
checkpoint = torch.load('model_checkpoints/best_classifier.pth', map_location=Config.DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(Config.DEVICE))
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = outputs.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

print("="*60)
print("CNN-ONLY RESULTS (threshold=0.5):")
print("="*60)
print(classification_report(all_labels, all_preds, target_names=['Benign', 'Malignant']))
print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
print(f"\nROC-AUC: {roc_auc_score(all_labels, all_probs):.4f}")