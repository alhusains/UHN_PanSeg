import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import numpy as np
import json
from pathlib import Path
from collections import Counter

class MultiTaskTrainer(nnUNetTrainer):
    # Custom trainer for multi-task learning with segmentation and classification
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        
        # Multi-task parameters
        self.num_classes_cls = 3
        self.cls_loss_weight = 10.0  # Higher weight to combat class imbalance
        
        # Load subtype labels and compute class weights
        self.subtype_info = self._load_subtype_info()
        self.class_weights = self._compute_class_weights()
        
        # Classification loss with weighted classes
        if self.class_weights is not None:
            self.cls_criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.cls_criterion = nn.CrossEntropyLoss()
        
        # Use focal loss for hard examples
        self.use_focal_loss = True
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0
        self.label_smoothing = 0.1

    def _load_subtype_info(self):
        # Load subtype information for classification labels
        subtype_file = Path('nnUNet_raw_data/Dataset001_Pancreas/subtype_info.json')
        if not subtype_file.exists():
            print("Warning: subtype_info.json not found")
            return {}
        
        with open(subtype_file, 'r') as f:
            return json.load(f)
    
    def _compute_class_weights(self):
        # Compute class weights to handle severe class imbalance
        if 'training' not in self.subtype_info:
            return None
        
        labels = list(self.subtype_info['training'].values())
        class_counts = Counter(labels)
        
        # Compute inverse frequency weights with additional scaling for minority classes
        total_samples = len(labels)
        num_classes = len(class_counts)
        
        class_weights = []
        for class_id in sorted(class_counts.keys()):
            count = class_counts[class_id]
            weight = total_samples / (num_classes * count)
            
            # Additional scaling for minority classes
            if count < total_samples / num_classes:
                weight *= 2.0
            
            class_weights.append(weight)
        
        # Normalize weights
        weight_sum = sum(class_weights)
        class_weights = [w * num_classes / weight_sum for w in class_weights]
        
        return torch.tensor(class_weights, dtype=torch.float32, device=self.device)

    def configure_optimizers(self):
        # Override to use standard PyTorch optimizers
        optimizer = torch.optim.SGD(
            self.network.parameters(), 
            lr=self.initial_lr, 
            weight_decay=self.weight_decay, 
            momentum=0.99
        )
        
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer, 
            total_iters=self.num_epochs, 
            power=0.9
        )
        
        return optimizer, lr_scheduler

    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import, 
                                 num_input_channels, num_output_channels, enable_deep_supervision: bool = True):
        # Build the multi-task network architecture with ResidualEncoderUNet
        
        from .src.models.nnunet_multitask import nnUNetv2MultiTask
        
        import torch.nn as nn
        
        # Setup network parameters
        network_kwargs = arch_init_kwargs.copy()
        network_kwargs['input_channels'] = num_input_channels
        network_kwargs['num_classes'] = num_output_channels
        network_kwargs['deep_supervision'] = enable_deep_supervision
        network_kwargs['num_classes_cls'] = 3
        
        # Convert string parameters to classes
        class_map = {
            'Conv3d': nn.Conv3d, 'Conv2d': nn.Conv2d,
            'InstanceNorm3d': nn.InstanceNorm3d, 'InstanceNorm2d': nn.InstanceNorm2d,
            'BatchNorm3d': nn.BatchNorm3d, 'BatchNorm2d': nn.BatchNorm2d,
            'LeakyReLU': nn.LeakyReLU, 'ReLU': nn.ReLU,
            'Dropout3d': nn.Dropout3d, 'Dropout2d': nn.Dropout2d, 'Dropout': nn.Dropout
        }
        
        for key in ['conv_op', 'norm_op', 'dropout_op', 'nonlin']:
            if key in network_kwargs and isinstance(network_kwargs[key], str):
                for name, cls in class_map.items():
                    if name in network_kwargs[key]:
                        network_kwargs[key] = cls
                        break
        
        return nnUNetv2MultiTask(**network_kwargs)

    def get_classification_labels(self, case_ids):
        # Get classification labels for given case IDs
        labels = []
        for case_id in case_ids:
            # Clean case ID
            base_id = case_id.replace('_0000', '')
            if base_id.startswith('val_'):
                base_id = base_id[4:]
            
            # Get label from training data
            if 'training' in self.subtype_info and base_id in self.subtype_info['training']:
                labels.append(self.subtype_info['training'][base_id])
            else:
                labels.append(0)  # Default class
        
        return torch.tensor(labels, dtype=torch.long, device=self.device)
    
    def focal_loss(self, inputs, targets, alpha=0.25, gamma=2.0):
        # Compute focal loss for handling hard examples
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    def label_smoothing_loss(self, inputs, targets, smoothing=0.1):
        # Apply label smoothing to classification loss
        confidence = 1.0 - smoothing
        log_probs = F.log_softmax(inputs, dim=1)
        
        true_class_loss = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=1)
        
        loss = confidence * true_class_loss + smoothing * smooth_loss
        return loss.mean()
    
    def enhance_contrast(self, data):
        # Apply contrast enhancement to medical images
        data_enhanced = data.clone()
        
        for i in range(data.shape[0]):
            img = data_enhanced[i, 0]
            
            # Clip to percentiles and apply gamma correction
            p1, p99 = torch.quantile(img, 0.01), torch.quantile(img, 0.99)
            img = torch.clamp(img, p1, p99)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = torch.pow(img, 0.8)  # Gamma correction
            data_enhanced[i, 0] = img * (p99 - p1) + p1
        
        return data_enhanced

    def train_step(self, batch: dict) -> dict:
        # Override nnU-Net's train_step for multi-task loss computation
        data = batch['data']
        target = batch['target']
        case_ids = batch['keys']  # Case IDs should always be present
        
        # Get classification labels
        cls_labels = self.get_classification_labels(case_ids)
        
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        cls_labels = cls_labels.to(self.device, non_blocking=True)
        
        # Apply contrast enhancement occasionally
        if np.random.random() < 0.3:
            data = self.enhance_contrast(data)
        
        self.optimizer.zero_grad(set_to_none=True)
        
        # Forward pass with autocast
        from torch import autocast
        from nnunetv2.utilities.helpers import dummy_context
        
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            seg_output = self.network(data)
            cls_output = self.network.last_classification_output
            
            # Compute losses
            seg_loss = self.loss(seg_output, target)
            
            if self.use_focal_loss:
                cls_loss = self.focal_loss(cls_output, cls_labels, self.focal_alpha, self.focal_gamma)
            else:
                cls_loss = self.label_smoothing_loss(cls_output, cls_labels, self.label_smoothing)
            
            total_loss = seg_loss + self.cls_loss_weight * cls_loss
            
            # Store losses for monitoring
            if not hasattr(self, 'seg_losses'):
                self.seg_losses = []
                self.cls_losses = []
                self.cls_accuracies = []
            
            self.seg_losses.append(seg_loss.item())
            self.cls_losses.append(cls_loss.item())
            
            # Compute accuracy
            with torch.no_grad():
                cls_pred = torch.argmax(cls_output, dim=1)
                cls_acc = (cls_pred == cls_labels).float().mean().item()
                self.cls_accuracies.append(cls_acc)
            
            # Periodic logging
            if len(self.seg_losses) % 50 == 0:
                print(f"Iter {len(self.seg_losses)}: Seg={seg_loss.item():.3f}, "
                      f"Cls={cls_loss.item():.3f}, Acc={cls_acc:.3f}")

        # Backward pass with gradient scaling
        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        
        return {'loss': total_loss.detach().cpu().numpy()}

    def validate(self, *args, **kwargs):
        # Override validation to handle multi-task evaluation
        val_loss = super().validate(*args, **kwargs)
        
        # Log recent classification performance
        if hasattr(self, 'cls_accuracies') and len(self.cls_accuracies) > 0:
            recent_cls_acc = np.mean(self.cls_accuracies[-50:])
            print(f"Recent classification accuracy: {recent_cls_acc:.3f}")
        
        return val_loss
    
    def on_epoch_end(self):
        # Enhanced epoch-end logging
        super().on_epoch_end()
        
        if hasattr(self, 'cls_accuracies') and len(self.cls_accuracies) > 0:
            recent_window = min(100, len(self.cls_accuracies))
            recent_acc = np.mean(self.cls_accuracies[-recent_window:])
            recent_cls_loss = np.mean(self.cls_losses[-recent_window:])
            
            print(f"Epoch {self.current_epoch}: Classification Acc={recent_acc:.3f}, Loss={recent_cls_loss:.4f}")
    
    def save_checkpoint(self, fname: str) -> None:
        # Override to save additional multi-task information
        super().save_checkpoint(fname)
        
        # Save multi-task metrics
        if hasattr(self, 'seg_losses'):
            checkpoint_data = {
                'seg_losses': self.seg_losses,
                'cls_losses': self.cls_losses,
                'cls_accuracies': getattr(self, 'cls_accuracies', []),
                'cls_loss_weight': self.cls_loss_weight,
                'num_classes_cls': self.num_classes_cls,
                'class_weights': self.class_weights.tolist() if self.class_weights is not None else None,
                'use_focal_loss': self.use_focal_loss,
                'label_smoothing': self.label_smoothing
            }
            
            multitask_fname = fname.replace('.pth', '_multitask.json')
            with open(multitask_fname, 'w') as f:
                json.dump(checkpoint_data, f, indent=2) 