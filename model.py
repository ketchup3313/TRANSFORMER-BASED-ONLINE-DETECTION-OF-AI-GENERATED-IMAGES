import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import random
import math
from pathlib import Path
import json
from datetime import datetime
from timm import create_model

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

class DCTLayer(nn.Module):
    """2D Discrete Cosine Transform layer"""
    def __init__(self, patch_size=8):
        super().__init__()
        self.patch_size = patch_size
        
        # Create DCT basis
        self.register_buffer('dct_basis', self._create_dct_basis())
        
    def _create_dct_basis(self):
        """Initialize DCT basis matrix"""
        p = self.patch_size
        dct_basis = torch.zeros(p, p)
        
        for u in range(p):
            for v in range(p):
                if u == 0:
                    Cu = 1.0 / math.sqrt(p)
                else:
                    Cu = math.sqrt(2.0 / p)
                    
                for x in range(p):
                    dct_basis[u, x] = Cu * math.cos((2 * x + 1) * u * math.pi / (2 * p))
                    
        return dct_basis
    
    def forward(self, x):
        """Apply 2D DCT to input features"""
        B, C, H, W = x.shape
        
        # Ensure H and W are divisible by patch_size
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            x = F.pad(x, (0, self.patch_size - W % self.patch_size, 
                          0, self.patch_size - H % self.patch_size))
            _, _, H, W = x.shape
        
        # Reshape into patches
        x_patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x_patches = x_patches.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        
        # Apply DCT
        dct_1 = torch.matmul(self.dct_basis.unsqueeze(0), x_patches)
        dct_2 = torch.matmul(dct_1.transpose(-2, -1), self.dct_basis.T.unsqueeze(0))
        
        # Reshape back
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        dct_2 = dct_2.view(B, C, num_patches_h, num_patches_w, self.patch_size, self.patch_size)
        dct_2 = dct_2.permute(0, 1, 2, 4, 3, 5).contiguous()
        dct_2 = dct_2.view(B, C, H, W)
        
        return dct_2

# %%
class ArtifactAttentionModule(nn.Module):
    """Artifact Attention Module (AAM) as described in the paper"""
    def __init__(self, in_channels, reduction_ratio=4):
        super().__init__()
        
        # Frequency branch with DCT
        self.dct_layer = DCTLayer(patch_size=8)
        self.freq_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // reduction_ratio),
            nn.GELU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels // reduction_ratio, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // reduction_ratio),
            nn.GELU()
        )
        
        # Spatial attention branch
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.BatchNorm2d(in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Multi-head attention for cross-attention
        self.num_heads = 4
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=self.num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Learnable fusion gate
        self.fusion_gate = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels + in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Frequency branch
        x_dct = self.dct_layer(x)
        freq_features = self.freq_conv(x_dct)
        
        # Spatial branch
        spatial_att = self.spatial_attention(x)
        spatial_features = x * spatial_att
        
        # Prepare for cross-attention
        spatial_flat = spatial_features.flatten(2).transpose(1, 2)  # B, HW, C
        
        # Upsample frequency features to match spatial resolution
        freq_up = F.interpolate(freq_features, size=(H, W), mode='bilinear', align_corners=False)
        freq_flat = freq_up.flatten(2).transpose(1, 2)  # B, HW, C/4
        
        # Pad frequency features to match channel dimension for cross-attention
        freq_flat_padded = F.pad(freq_flat, (0, C - freq_flat.size(-1)))
        
        # Cross-attention: spatial features attend to frequency features
        attended_features, _ = self.cross_attention(
            query=spatial_flat,
            key=freq_flat_padded,
            value=freq_flat_padded
        )
        attended_features = attended_features.transpose(1, 2).reshape(B, C, H, W)
        
        # Adaptive fusion with learnable gate
        gate = torch.sigmoid(self.fusion_gate)
        
        # Concatenate attended features and upsampled frequency features
        combined = torch.cat([attended_features, freq_up], dim=1)
        fused = self.fusion(combined)
        
        # Residual connection
        output = x + fused
        
        return output

# %%
class AAMSwinTransformer(nn.Module):
    """Swin Transformer with integrated AAM modules"""
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        
        # Load base Swin Transformer
        self.backbone = create_model('swin_base_patch4_window7_224', pretrained=pretrained)
        
        # Get feature dimensions from each stage
        embed_dims = [128, 256, 512, 1024]
        
        # Create AAM modules for each stage
        self.aam_modules = nn.ModuleList([
            ArtifactAttentionModule(dim) for dim in embed_dims
        ])
        
        # Replace the classification head
        num_features = self.backbone.num_features
        self.backbone.head = nn.Identity()
        
        # New classification head with uncertainty estimation
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Uncertainty head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward_features(self, x):
        """Forward through backbone with AAM integration"""
        # Patch embedding
        x = self.backbone.patch_embed(x)
        if self.backbone.absolute_pos_embed is not None:
            x = x + self.backbone.absolute_pos_embed
        x = self.backbone.pos_drop(x)
        
        # Process through stages with AAM
        for i, layer in enumerate(self.backbone.layers):
            # Get spatial dimensions for current stage
            H, W = layer.input_resolution
            B, L, C = x.shape
            
            # Process through Swin layer
            x = layer(x)
            
            # Reshape to 2D for AAM
            x_2d = x.transpose(1, 2).reshape(B, C, H, W)
            
            # Apply AAM
            x_2d = self.aam_modules[i](x_2d)
            
            # Reshape back to sequence
            x = x_2d.flatten(2).transpose(1, 2)
            
        # Final norm
        x = self.backbone.norm(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        return x
        
    def forward(self, x):
        features = self.forward_features(x)
        
        # Classification
        logits = self.classifier(features)
        
        # Uncertainty estimation
        uncertainty = self.uncertainty_head(features)
        
        return {
            'logits': logits,
            'uncertainty': uncertainty,
            'features': features
        }

# %% [markdown]
# ## 3. Loss Functions and Data Augmentation

# %%
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p = torch.exp(-ce_loss)
        loss = self.alpha * (1 - p) ** self.gamma * ce_loss
        return loss.mean()


class DegradationAugmentation:
    """Apply realistic degradations during training"""
    def __init__(self, p_degrade=0.5):
        self.p_degrade = p_degrade
        
    def jpeg_compression(self, img, quality=None):
        """Simulate JPEG compression"""
        if quality is None:
            quality = np.random.randint(30, 95)
        
        # Simplified JPEG simulation using noise
        noise = torch.randn_like(img) * (0.1 * (100 - quality) / 100)
        return torch.clamp(img + noise, 0, 1)
    
    def gaussian_blur(self, img, sigma=None):
        """Apply Gaussian blur"""
        if sigma is None:
            sigma = np.random.uniform(0.5, 5.0)
            
        kernel_size = int(2 * np.ceil(3 * sigma) + 1)
        
        # Create Gaussian kernel
        x = torch.arange(kernel_size, dtype=torch.float32, device=img.device) - kernel_size // 2
        gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        kernel = gauss / gauss.sum()
        kernel = kernel.view(1, 1, kernel_size) * kernel.view(1, kernel_size, 1)
        
        # Apply blur
        C = img.shape[0]
        kernel = kernel.repeat(C, 1, 1, 1)
        img = F.pad(img.unsqueeze(0), (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2), mode='reflect')
        img = F.conv2d(img, kernel, groups=C)
        
        return img.squeeze(0)
    
    def resolution_reduction(self, img, scale=None):
        """Reduce and restore resolution"""
        if scale is None:
            scale = np.random.choice([0.25, 0.5, 0.75])
            
        _, H, W = img.shape
        low_res = F.interpolate(img.unsqueeze(0), scale_factor=scale, mode='bilinear', align_corners=False)
        restored = F.interpolate(low_res, size=(H, W), mode='bilinear', align_corners=False)
        
        return restored.squeeze(0)
    
    def __call__(self, img):
        """Apply random degradation with probability p_degrade"""
        if np.random.random() < self.p_degrade:
            degradation_type = np.random.choice(['jpeg', 'blur', 'resolution', 'combined'])
            
            if degradation_type == 'jpeg':
                return self.jpeg_compression(img)
            elif degradation_type == 'blur':
                return self.gaussian_blur(img)
            elif degradation_type == 'resolution':
                return self.resolution_reduction(img)
            else:  # combined
                img = self.jpeg_compression(img)
                if np.random.random() < 0.5:
                    img = self.gaussian_blur(img)
                if np.random.random() < 0.5:
                    img = self.resolution_reduction(img)
                return img
        
        return img

# %% [markdown]
# ## 4. Dataset Class

# %%
class AIGeneratedDataset(Dataset):
    """Dataset for AI-generated image detection"""
    def __init__(self, root_dir, transform=None, degradation=None, is_training=False):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.degradation = degradation
        self.is_training = is_training
        
        # Collect all images
        self.images = []
        self.labels = []
        
        # Real images (label=0)
        real_dir = self.root_dir / 'REAL'
        if real_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                for img_path in real_dir.glob(ext):
                    self.images.append(str(img_path))
                    self.labels.append(0)
            
        # Fake images (label=1)
        fake_dir = self.root_dir / 'FAKE'
        if fake_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                for img_path in fake_dir.glob(ext):
                    self.images.append(str(img_path))
                    self.labels.append(1)
            
        print(f"Dataset {root_dir}: {len(self.images)} images "
              f"({sum(l==0 for l in self.labels)} real, {sum(l==1 for l in self.labels)} fake)")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        # Apply degradation (only during training)
        if self.degradation is not None and self.is_training:
            image = self.degradation(image)
            
        return image, label

# %% [markdown]
# ## 5. Training Configuration

# %%
# Configuration
class Config:
    # Data settings
    data_dir = './data'  # Change this to your dataset path
    output_dir = './outputs'
    
    # Model settings
    model_name = 'aam_swin'
    pretrained = True
    
    # Training settings
    batch_size = 16  # Adjust based on your GPU memory
    num_workers = 4
    
    # Phase-specific epochs
    phase1_epochs = 20  # Reduced for notebook demo
    phase2_epochs = 30
    phase3_epochs = 10
    
    # Optimization settings
    lr = 1e-4
    weight_decay = 0.05
    
    # Hardware settings
    gpu = 0
    seed = 42
    
    # Logging
    save_interval = 5
    log_interval = 10

config = Config()

# Create output directory
Path(config.output_dir).mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 6. Three-Phase Training Implementation

# %%
class ThreePhaseTrainer:
    """Three-phase training strategy as described in the paper"""
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = device
        self.model.to(self.device)
        
        # Loss functions
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Metrics
        self.best_val_acc = 0
        self.phase = 1
        
        # Create save directory
        self.save_dir = Path(config.output_dir) / config.model_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
    def setup_data_loaders(self):
        """Setup data loaders with appropriate transforms"""
        # Base transforms
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Training transforms with augmentation
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            normalize,
        ])
        
        # Validation transforms (no augmentation)
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
        
        # Degradation augmentation for phase 2 and 3
        degradation = DegradationAugmentation(p_degrade=0.5) if self.phase >= 2 else None
        
        # Create datasets
        train_dataset = AIGeneratedDataset(
            os.path.join(self.config.data_dir, 'train'),
            transform=train_transform,
            degradation=degradation,
            is_training=True
        )
        
        val_dataset = AIGeneratedDataset(
            os.path.join(self.config.data_dir, 'test'),
            transform=val_transform,
            is_training=False
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        return self.train_loader, self.val_loader
    
    def setup_optimizer_phase1(self):
        """Phase 1: Train only AAM modules"""
        # Freeze backbone, train only AAM
        for param in self.model.backbone.parameters():
            param.requires_grad = False
        
        for param in self.model.aam_modules.parameters():
            param.requires_grad = True
            
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.phase1_epochs
        )
        
    def setup_optimizer_phase2(self):
        """Phase 2: Train full model with lower learning rate"""
        # Unfreeze all parameters
        for param in self.model.parameters():
            param.requires_grad = True
            
        self.optimizer = optim.AdamW([
            {'params': self.model.backbone.parameters(), 'lr': self.config.lr * 0.1},
            {'params': self.model.aam_modules.parameters(), 'lr': self.config.lr},
            {'params': self.model.classifier.parameters(), 'lr': self.config.lr},
            {'params': self.model.uncertainty_head.parameters(), 'lr': self.config.lr}
        ], weight_decay=self.config.weight_decay)
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
    def setup_optimizer_phase3(self):
        """Phase 3: Fine-tuning with adversarial training"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr * 0.01,
            weight_decay=self.config.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5, gamma=0.5
        )
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Train Epoch {epoch} (Phase {self.phase})')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            
            # Compute loss based on phase
            if self.phase == 1:
                loss = self.ce_loss(outputs['logits'], labels)
            elif self.phase == 2:
                loss = self.focal_loss(outputs['logits'], labels)
            else:  # Phase 3
                loss = self.ce_loss(outputs['logits'], labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = outputs['logits'].max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
        
        return total_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Val Epoch {epoch}')
            
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.ce_loss(outputs['logits'], labels)
                
                total_loss += loss.item()
                _, predicted = outputs['logits'].max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100. * correct / total:.2f}%'
                })
        
        val_loss = total_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'phase': self.phase,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'config': vars(self.config),
            'history': self.history
        }
        
        checkpoint_path = self.save_dir / f'checkpoint_phase{self.phase}_epoch{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.save_dir / f'best_model_phase{self.phase}.pth'
            torch.save(checkpoint, best_path)
            print(f"New best model saved with accuracy: {val_acc:.2f}%")
    
    def plot_training_history(self):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Acc')
        plt.plot(self.history['val_acc'], label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_history.png')
        plt.show()
    
    def train(self):
        """Complete three-phase training"""
        total_epochs = 0
        
        # Phase 1: Artifact-specific pretraining
        print("\n=== Phase 1: Artifact-Specific Pretraining ===")
        self.phase = 1
        self.setup_data_loaders()
        self.setup_optimizer_phase1()
        
        phase1_best_acc = 0
        for epoch in range(1, self.config.phase1_epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)
            self.scheduler.step()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save checkpoint
            if val_acc > phase1_best_acc:  # 修复：原来是 phase2_best_acc
                phase1_best_acc = val_acc
                self.save_checkpoint(epoch, val_acc, is_best=True)
        
        # Phase 2: Degradation-aware joint training
        print("\n=== Phase 2: Degradation-Aware Joint Training ===")
        self.phase = 2
        self.setup_data_loaders()  # Reload with degradation
        self.setup_optimizer_phase2()
        
        phase2_best_acc = 0
        for epoch in range(1, self.config.phase2_epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)
            self.scheduler.step()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save checkpoint
            if val_acc > phase2_best_acc:
                phase2_best_acc = val_acc
                self.save_checkpoint(epoch, val_acc, is_best=True)
        
        # Phase 3: Adversarial robustness tuning
        print("\n=== Phase 3: Adversarial Robustness Tuning ===")
        self.phase = 3
        self.setup_optimizer_phase3()
        
        phase3_best_acc = 0
        for epoch in range(1, self.config.phase3_epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)
            self.scheduler.step()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save checkpoint
            if val_acc > phase3_best_acc:
                phase3_best_acc = val_acc
                self.save_checkpoint(epoch, val_acc, is_best=True)
        
        print("\n=== Training Complete ===")
        print(f"Best accuracy - Phase 1: {phase1_best_acc:.2f}%, Phase 2: {phase2_best_acc:.2f}%, Phase 3: {phase3_best_acc:.2f}%")
        
        # Plot training history
        self.plot_training_history()
        
        # Save final model
        final_checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': vars(self.config),
            'best_acc': {
                'phase1': phase1_best_acc,
                'phase2': phase2_best_acc,
                'phase3': phase3_best_acc
            },
            'history': self.history
        }
        torch.save(final_checkpoint, self.save_dir / 'final_model.pth')
        
        return self.history

print("Creating AAM-Swin Transformer model...")
model = AAMSwinTransformer(num_classes=2, pretrained=config.pretrained)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

trainer = ThreePhaseTrainer(model, config)

# Uncomment the line below to start training
# history = trainer.train()

print("Training setup complete! Uncomment the line above to start training.")

# %% [markdown]
# ## 9. Model Inference and Evaluation

# %%
def load_model_for_inference(checkpoint_path, device):
    """Load trained model for inference"""
    model = AAMSwinTransformer(num_classes=2, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def predict_single_image(model, image_path, device):
    """Predict whether a single image is AI-generated"""
    # Prepare transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs['logits'], dim=1)
        uncertainty = outputs['uncertainty'].item()
        
    # Get prediction
    fake_prob = probs[0, 1].item()
    is_fake = fake_prob > 0.5
    
    return {
        'is_ai_generated': is_fake,
        'ai_probability': fake_prob,
        'real_probability': probs[0, 0].item(),
        'uncertainty': uncertainty,
        'confidence': 1 - uncertainty
    }



# %%
def visualize_predictions(model, data_loader, num_images=8, device='cuda'):
    """Visualize model predictions on sample images"""
    model.eval()
    
    images, labels, predictions = [], [], []
    
    with torch.no_grad():
        for batch_images, batch_labels in data_loader:
            batch_images = batch_images.to(device)
            outputs = model(batch_images)
            probs = F.softmax(outputs['logits'], dim=1)
            preds = probs[:, 1] > 0.5  # Probability of being fake
            
            images.extend(batch_images.cpu())
            labels.extend(batch_labels)
            predictions.extend(preds.cpu())
            
            if len(images) >= num_images:
                break
    
    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # Plot
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.ravel()
    
    for i in range(min(num_images, len(images))):
        img = images[i] * std + mean
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0).numpy()
        
        axes[i].imshow(img)
        axes[i].axis('off')
        
        true_label = 'Fake' if labels[i] == 1 else 'Real'
        pred_label = 'Fake' if predictions[i] else 'Real'
        color = 'green' if (labels[i] == 1) == predictions[i] else 'red'
        
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}', color=color)
    
    plt.tight_layout()
    plt.show()



def export_model_for_deployment(model, save_path, example_input_shape=(1, 3, 224, 224)):
    """Export model for deployment (ONNX format)"""
    model.eval()

    example_input = torch.randn(example_input_shape).to(device)
    
    torch.onnx.export(
        model,
        example_input,
        save_path,
        input_names=['input'],
        output_names=['logits', 'uncertainty', 'features'],
        dynamic_axes={'input': {0: 'batch_size'}},
        opset_version=11
    )
    
    print(f"Model exported to {save_path}")