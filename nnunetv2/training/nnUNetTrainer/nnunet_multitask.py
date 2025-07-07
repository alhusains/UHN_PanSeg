import torch
import torch.nn as nn
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

class nnUNetv2MultiTask(ResidualEncoderUNet):
    """Multi-task nnU-Net v2 model for segmentation and classification."""
    
    def __init__(self, input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                 n_conv_per_stage, n_conv_per_stage_decoder, conv_bias, norm_op, norm_op_kwargs,
                 dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, deep_supervision, num_classes,
                 num_classes_cls=3, **kwargs):
        
        # Map nnU-Net parameter names to ResidualEncoderUNet
        n_blocks_per_stage = n_conv_per_stage
        
        # Initialize parent ResidualEncoderUNet
        super().__init__(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_blocks_per_stage=n_blocks_per_stage,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision
        )
        
        self.num_classes_cls = num_classes_cls
        self.features_per_stage = features_per_stage
        
        # Feature extraction setup - ORIGINAL MULTI-STAGE APPROACH
        self._multi_stage_features = {}
        self._hook_handles = []
        
        # Use last 3 encoder stages for better feature representation
        self.stages_to_use = min(3, len(features_per_stage))
        total_features = sum(features_per_stage[-self.stages_to_use:]) * 2  # *2 for dual pooling
        
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.global_max_pool = nn.AdaptiveMaxPool3d(1)
        
        # Classification head with multi-stage features - ORIGINAL ARCHITECTURE
        self.classification_head = nn.Sequential(
            nn.BatchNorm1d(total_features),
            nn.Dropout(0.3),
            nn.Linear(total_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes_cls)
        )
        
        self.last_classification_output = None
        
        # Register feature extraction hooks
        self._register_multi_stage_hooks()
    
    def _register_multi_stage_hooks(self):
        """Register hooks on multiple encoder stages for feature extraction."""
        if not (hasattr(self.encoder, 'stages') and len(self.encoder.stages) > 0):
            return
        
        # Hook the last few encoder stages
        stages_to_hook = self.encoder.stages[-self.stages_to_use:]
        
        for i, stage in enumerate(stages_to_hook):
            stage_idx = len(self.encoder.stages) - self.stages_to_use + i
            
            def create_hook(stage_id):
                def hook_fn(module, input, output):
                    self._multi_stage_features[stage_id] = output
                return hook_fn
            
            handle = stage.register_forward_hook(create_hook(stage_idx))
            self._hook_handles.append(handle)
    
    def _process_multi_stage_features(self):
        """Process features from multiple encoder stages."""
        if not self._multi_stage_features:
            return None
        
        processed_features = []
        feature_stats = {}
        
        # Process each hooked stage
        for stage_idx in sorted(self._multi_stage_features.keys()):
            features = self._multi_stage_features[stage_idx]
            
            # Apply pooling
            avg_pooled = self.global_avg_pool(features).view(features.size(0), -1)
            max_pooled = self.global_max_pool(features).view(features.size(0), -1)
            
            # L2 normalize to prevent scale issues
            avg_pooled = nn.functional.normalize(avg_pooled, p=2, dim=1)
            max_pooled = nn.functional.normalize(max_pooled, p=2, dim=1)
            
            # Combine avg and max pooling
            stage_features = torch.cat([avg_pooled, max_pooled], dim=1)
            processed_features.append(stage_features)
            
            # Collect stats for diagnostics
            feature_stats[f'stage_{stage_idx}'] = {
                'mean': features.mean().item(),
                'std': features.std().item(),
                'min': features.min().item(),
                'max': features.max().item(),
                'shape': list(features.shape)
            }
        
        # Concatenate all stage features
        combined_features = torch.cat(processed_features, dim=1)
        self.last_feature_stats = feature_stats
        
        return combined_features
    
    def forward(self, x):
        """Forward pass with multi-stage feature capture."""
        self._multi_stage_features.clear()
        
        # Use parent's forward for segmentation
        seg_output = super().forward(x)
        
        # Process classification features
        combined_features = self._process_multi_stage_features()
        
        if combined_features is not None:
            cls_output = self.classification_head(combined_features)
            self.last_classification_output = cls_output
        else:
            # Fallback: dummy classification output
            batch_size = x.size(0)
            self.last_classification_output = torch.zeros(batch_size, self.num_classes_cls, device=x.device)
        
        return seg_output
    
    def forward_segmentation_only(self, x):
        """Forward pass for segmentation only."""
        return super().forward(x)
    
    def forward_classification_only(self, x):
        """Forward pass for classification only."""
        self._multi_stage_features.clear()
        
        # Run through encoder stages
        current = x
        for i, encoder_stage in enumerate(self.encoder.stages):
            current = encoder_stage(current)
        
        # Process features for classification
        combined_features = self._process_multi_stage_features()
        
        if combined_features is not None:
            return self.classification_head(combined_features)
        else:
            raise ValueError("No multi-stage features captured for classification")
    
    def get_feature_diagnostics(self):
        """Get diagnostic information about extracted features."""
        if hasattr(self, 'last_feature_stats'):
            return self.last_feature_stats
        return {}
    
    def __del__(self):
        """Clean up hooks when object is destroyed."""
        for handle in self._hook_handles:
            if handle is not None:
                handle.remove() 