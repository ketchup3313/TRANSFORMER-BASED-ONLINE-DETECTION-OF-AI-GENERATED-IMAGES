import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import wandb
from pathlib import Path
import json
from datetime import datetime
import argparse
from torch.cuda.amp import GradScaler, autocast

from model import AAMSwinTransformer, FocalLoss, DegradationAugmentation


class AIGeneratedDataset(Dataset):
    """Dataset for AI-generated image detection"""
    def __init__(self, root_dir, transform=None, degradation=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.degradation = degradation
        
        self.images = []
        self.labels = []
        
        real_dir = self.root_dir / 'REAL'
        for img_path in real_dir.glob('*.jpg'):
            self.images.append(str(img_path))
            self.labels.append(0)
        for img_path in real_dir.glob('*.png'):
            self.images.append(str(img_path))
            self.labels.append(0)
            
        fake_dir = self.root_dir / 'FAKE'
        for img_path in fake_dir.glob('*.jpg'):
            self.images.append(str(img_path))
            self.labels.append(1)
        for img_path in fake_dir.glob('*.png'):
            self.images.append(str(img_path))
            self.labels.append(1)
            
        print(f"Dataset {root_dir}: {len(self.images)} images "
              f"({sum(l==0 for l in self.labels)} real, {sum(l==1 for l in self.labels)} fake)")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        if self.degradation is not None and self.training:
            image = self.degradation(image)
            
        return image, label


class ThreePhaseTrainer:
    """Three-phase training strategy as described in the paper"""
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        if config.use_wandb:
            wandb.init(project="ai-image-detection", config=config)
            wandb.watch(model)
        
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        
        self.best_val_acc = 0
        self.phase = 1
        
    def setup_data_loaders(self):

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.33))
        ])

        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

        degradation = DegradationAugmentation(p_degrade=0.5) if self.phase >= 2 else None

        train_dataset = AIGeneratedDataset(
            os.path.join(self.config.data_dir, 'train'),
            transform=train_transform,
            degradation=degradation
        )
        train_dataset.training = True
        
        val_dataset = AIGeneratedDataset(
            os.path.join(self.config.data_dir, 'test'),
            transform=val_transform
        )
        val_dataset.training = False
        
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
            lr=1e-4,
            weight_decay=0.05
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.phase1_epochs
        )
        
    def setup_optimizer_phase2(self):

        for param in self.model.parameters():
            param.requires_grad = True
            
        self.optimizer = optim.AdamW([
            {'params': self.model.backbone.parameters(), 'lr': 5e-5},
            {'params': self.model.aam_modules.parameters(), 'lr': 1e-4},
            {'params': self.model.classifier.parameters(), 'lr': 1e-4},
            {'params': self.model.uncertainty_head.parameters(), 'lr': 1e-4}
        ], weight_decay=0.05)
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
    def setup_optimizer_phase3(self):
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-5,
            weight_decay=0.05
        )
        
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5, gamma=0.5
        )
    
    def compute_phase1_loss(self, outputs, labels):
        logits = outputs['logits']
        

        cls_loss = self.ce_loss(logits, labels)
        

        features = outputs['features']
        feature_mean = features.mean(dim=0)
        feature_std = features.std(dim=0)
        

        diversity_loss = -torch.log(feature_std.mean() + 1e-6)
        
        total_loss = cls_loss + 0.1 * diversity_loss
        
        return total_loss, {'cls_loss': cls_loss.item(), 'diversity_loss': diversity_loss.item()}
    
    def compute_phase2_loss(self, outputs, labels, outputs_clean=None):
        """Phase 2: Focal loss + Consistency loss"""
        logits = outputs['logits']
        

        focal_loss = self.focal_loss(logits, labels)

        consistency_loss = 0
        if outputs_clean is not None:
            probs = F.softmax(logits, dim=1)
            probs_clean = F.softmax(outputs_clean['logits'], dim=1)
            consistency_loss = F.kl_div(
                probs.log(), probs_clean, reduction='batchmean'
            )
        
        total_loss = focal_loss + 0.3 * consistency_loss
        
        return total_loss, {
            'focal_loss': focal_loss.item(),
            'consistency_loss': consistency_loss.item() if outputs_clean else 0
        }
    
    def compute_phase3_loss(self, outputs, labels, adv_outputs=None):
        """Phase 3: Adversarial robustness loss"""
        logits = outputs['logits']
        
        cls_loss = self.ce_loss(logits, labels)
        
        adv_loss = 0
        if adv_outputs is not None:
            adv_loss = self.ce_loss(adv_outputs['logits'], labels)
        
        total_loss = cls_loss + 0.5 * adv_loss
        
        return total_loss, {
            'cls_loss': cls_loss.item(),
            'adv_loss': adv_loss.item() if adv_outputs else 0
        }
    
    def generate_adversarial_examples(self, images, labels, epsilon=0.03):
        """Generate adversarial examples using FGSM"""
        images.requires_grad = True
        
        outputs = self.model(images)
        loss = self.ce_loss(outputs['logits'], labels)
        
        self.model.zero_grad()
        loss.backward()

        adv_images = images + epsilon * images.grad.sign()
        adv_images = torch.clamp(adv_images, 0, 1)
        
        return adv_images.detach()
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Train Epoch {epoch} (Phase {self.phase})')
        
        scaler = GradScaler()
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast():
                outputs = self.model(images)
                
                if self.phase == 1:
                    loss, loss_dict = self.compute_phase1_loss(outputs, labels)
                elif self.phase == 2:
                    with torch.no_grad():
                        outputs_clean = self.model(images.clone())
                    loss, loss_dict = self.compute_phase2_loss(outputs, labels, outputs_clean)
                else:  # Phase 3
                    # Generate adversarial examples
                    if batch_idx % 2 == 0:  # Apply adversarial training every other batch
                        adv_images = self.generate_adversarial_examples(images, labels)
                        with autocast():
                            adv_outputs = self.model(adv_images)
                        loss, loss_dict = self.compute_phase3_loss(outputs, labels, adv_outputs)
                    else:
                        loss, loss_dict = self.compute_phase3_loss(outputs, labels)
            
            # Backward pass with mixed precision
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = outputs['logits'].max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / total,
                **loss_dict
            })
            
            # Log to wandb
            if self.config.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    f'train/loss_phase{self.phase}': loss.item(),
                    f'train/acc_phase{self.phase}': 100. * correct / total,
                    **{f'train/{k}_phase{self.phase}': v for k, v in loss_dict.items()}
                })
        
        return total_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        all_probs = []
        all_labels = []
        all_uncertainties = []
        
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
                
                # Collect predictions for analysis
                probs = F.softmax(outputs['logits'], dim=1)
                all_probs.append(probs.cpu())
                all_labels.append(labels.cpu())
                all_uncertainties.append(outputs['uncertainty'].cpu())
                
                pbar.set_postfix({
                    'loss': loss.item(),
                    'acc': 100. * correct / total
                })
        
        val_loss = total_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        

        all_probs = torch.cat(all_probs)
        all_labels = torch.cat(all_labels)
        all_uncertainties = torch.cat(all_uncertainties)

        if self.config.use_wandb:
            wandb.log({
                f'val/loss_phase{self.phase}': val_loss,
                f'val/acc_phase{self.phase}': val_acc,
                f'val/uncertainty_mean_phase{self.phase}': all_uncertainties.mean().item()
            })
        
        return val_loss, val_acc
    
    def train(self):
        """Complete three-phase training"""
        print("\n=== Phase 1: Artifact-Specific Pretraining ===")
        self.phase = 1
        self.setup_data_loaders()
        self.setup_optimizer_phase1()
        
        for epoch in range(1, self.config.phase1_epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)