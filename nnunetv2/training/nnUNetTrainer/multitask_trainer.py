import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import numpy as np
import json
from pathlib import Path
from collections import Counter
import pandas as pd
from sklearn.metrics import f1_score

class MultiTaskTrainer(nnUNetTrainer):
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        
        #multi-task parameters
        self.num_classes_cls = 3
        self.cls_loss_weight = 0.3
        
        #load subtype labels
        self.subtype_dict = self._load_subtype_info()
        
        #simple cross-entropy loss
        self.cls_criterion = nn.CrossEntropyLoss()
        
        print(f"Classification loss weight: {self.cls_loss_weight}")

    def _load_subtype_info(self):
        subtype_file = Path('nnUNet_preprocessed/Dataset001_Pancreas/subtype_info.json')
        if subtype_file.exists():
            with open(subtype_file, 'r') as f:
                return json.load(f)
        
        #try alternative location
        subtype_file = Path('nnUNet_preprocessed/Dataset001_Pancreas/subtype_results_train.csv')
        if subtype_file.exists():
            df = pd.read_csv(subtype_file)
            return {
                'training': {
                    row['Name'].replace(".nii.gz", ""): int(row['Subtype'])
                    for _, row in df.iterrows()
                }
            }
        
        print("Warning: No subtype info found, using default labels")
        return {'training': {}}

    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import, 
                                 num_input_channels, num_output_channels, enable_deep_supervision: bool = True):
        
        from .src.models.nnunet_multitask import nnUNetv2MultiTask
        
        import torch.nn as nn
        
        #setup network parameters
        network_kwargs = arch_init_kwargs.copy()
        network_kwargs['input_channels'] = num_input_channels
        network_kwargs['num_classes'] = num_output_channels
        network_kwargs['deep_supervision'] = enable_deep_supervision
        network_kwargs['num_classes_cls'] = 3
        
        #convert string parameters to classes
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
        labels = []
        for case_id in case_ids:
            #clean case ID
            base_id = case_id.replace('_0000', '')
            if base_id.startswith('val_'):
                base_id = base_id[4:]
            
            #get label from training data
            if 'training' in self.subtype_dict and base_id in self.subtype_dict['training']:
                labels.append(self.subtype_dict['training'][base_id])
            else:
                labels.append(0)
        
        return torch.tensor(labels, dtype=torch.long, device=self.device)

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        case_ids = batch['keys']
        
        #get classification labels
        cls_labels = self.get_classification_labels(case_ids)
        
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        cls_labels = cls_labels.to(self.device, non_blocking=True)
        
        self.optimizer.zero_grad(set_to_none=True)
        
        #forward pass with autocast
        from torch import autocast
        from nnunetv2.utilities.helpers import dummy_context
        
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            #get segmentation output
            seg_output = self.network(data)
            
            #get classification output directly from encoder features
            enc_features = self.network.encoder(data)
            cls_output = self.network.ClassificationHead(enc_features)
            
            #compute losses
            seg_loss = self.loss(seg_output, target)
            cls_loss = self.cls_criterion(cls_output, cls_labels)
            
            #simple weighted combination
            total_loss = seg_loss + self.cls_loss_weight * cls_loss
            
            #store for monitoring
            if not hasattr(self, 'seg_losses'):
                self.seg_losses = []
                self.cls_losses = []
                self.cls_accuracies = []
            
            self.seg_losses.append(seg_loss.item())
            self.cls_losses.append(cls_loss.item())
            
            #compute accuracy
            with torch.no_grad():
                cls_pred = torch.argmax(cls_output, dim=1)
                cls_acc = (cls_pred == cls_labels).float().mean().item()
                self.cls_accuracies.append(cls_acc)
            
            #periodic logging
            if len(self.seg_losses) % 50 == 0:
                print(f"Iter {len(self.seg_losses)}: Seg={seg_loss.item():.3f}, "
                      f"Cls={cls_loss.item():.3f}, Acc={cls_acc:.3f}")

        #backward pass
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

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        case_ids = batch['keys']
        
        #get classification labels
        cls_labels = self.get_classification_labels(case_ids)
        
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        cls_labels = cls_labels.to(self.device, non_blocking=True)
        
        from torch import autocast
        from nnunetv2.utilities.helpers import dummy_context
        
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            #get segmentation output
            seg_output = self.network(data)
            
            #get classification output directly from encoder features
            enc_features = self.network.encoder(data)
            cls_output = self.network.ClassificationHead(enc_features)
            
            #compute losses
            seg_loss = self.loss(seg_output, target)
            cls_loss = self.cls_criterion(cls_output, cls_labels)
            total_loss = seg_loss + self.cls_loss_weight * cls_loss
            
            #compute F1 score
            preds = torch.argmax(cls_output, dim=1)
            f1 = f1_score(cls_labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
        
        #standard nnU-Net validation metrics
        from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
        
        if self.enable_deep_supervision:
            seg_output = seg_output[0]
            target = target[0]

        axes = [0] + list(range(2, seg_output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(seg_output) > 0.5).long()
        else:
            output_seg = seg_output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(seg_output.shape, device=seg_output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:] if target.dtype != torch.bool else ~target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)
        
        return {
            'loss': total_loss.detach().cpu().numpy(),
            'classification_f1': f1,
            'tp_hard': tp.detach().cpu().numpy(),
            'fp_hard': fp.detach().cpu().numpy(),
            'fn_hard': fn.detach().cpu().numpy()
        }

    def on_validation_epoch_end(self, val_outputs):
        #call parent method for segmentation metrics
        super().on_validation_epoch_end(val_outputs)
        
        #compute classification F1
        if hasattr(self, 'cls_accuracies') and len(self.cls_accuracies) > 0:
            recent_acc = np.mean(self.cls_accuracies[-50:])
            print(f"Recent classification accuracy: {recent_acc:.3f}")
        
        #compute F1 from validation outputs
        if len(val_outputs) > 0 and 'classification_f1' in val_outputs[0]:
            f1_scores = [out['classification_f1'] for out in val_outputs]
            mean_f1 = np.mean(f1_scores)
            
            #store F1 in logger
            if not hasattr(self.logger, 'classification_f1'):
                self.logger.classification_f1 = []
            self.logger.classification_f1.append(mean_f1)
            
            print(f"Validation Classification F1: {mean_f1:.3f}")

    def on_epoch_end(self):
        super().on_epoch_end()
        
        #log recent classification performance
        if hasattr(self, 'cls_accuracies') and len(self.cls_accuracies) > 0:
            recent_window = min(100, len(self.cls_accuracies))
            recent_acc = np.mean(self.cls_accuracies[-recent_window:])
            recent_cls_loss = np.mean(self.cls_losses[-recent_window:])
            
            print(f"Epoch {self.current_epoch}: Classification Acc={recent_acc:.3f}, Loss={recent_cls_loss:.4f}")
        
        #save best model based on combined score
        if hasattr(self.logger, 'classification_f1') and len(self.logger.classification_f1) > 0:
            f1 = self.logger.classification_f1[-1]
            
            #get segmentation performance
            if hasattr(self.logger, 'my_fantastic_logging') and 'dice_per_class_or_region' in self.logger.my_fantastic_logging:
                dice_per_class = self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]
                whole_pancreas_dsc = np.mean(dice_per_class[1:])  # labels 1 and 2
                
                #combined score
                combined_score = (whole_pancreas_dsc + f1) / 2
                
                #initialize if not yet done
                if not hasattr(self, '_best_combined_score'):
                    self._best_combined_score = -np.inf
                
                #save best model
                if combined_score > self._best_combined_score and whole_pancreas_dsc > 0.7 and f1 > 0.6:
                    self._best_combined_score = combined_score
                    print(f"New best model: whole_dsc={whole_pancreas_dsc:.4f}, f1={f1:.4f}, combined={combined_score:.4f}")
                    
                    #save checkpoint
                    from batchgenerators.utilities.file_and_folder_operations import join
                    self.save_checkpoint(join(self.output_folder, 'checkpoint_best_combined.pth'))

    def save_checkpoint(self, fname: str) -> None:
        super().save_checkpoint(fname)
        
        #save multi-task metrics
        if hasattr(self, 'seg_losses'):
            checkpoint_data = {
                'seg_losses': self.seg_losses,
                'cls_losses': self.cls_losses,
                'cls_accuracies': getattr(self, 'cls_accuracies', []),
                'cls_loss_weight': self.cls_loss_weight,
                'num_classes_cls': self.num_classes_cls,
                'approach': 'successful_implementation'
            }
            
            multitask_fname = fname.replace('.pth', '_multitask.json')
            with open(multitask_fname, 'w') as f:
                json.dump(checkpoint_data, f, indent=2) 