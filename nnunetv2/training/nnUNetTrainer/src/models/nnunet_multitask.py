import torch
import torch.nn as nn
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

class nnUNetv2MultiTask(ResidualEncoderUNet):
    """Multi-task nnU-Net v2 model for segmentation and classification."""
    
    def __init__(self, input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                 n_conv_per_stage, n_conv_per_stage_decoder, conv_bias, norm_op, norm_op_kwargs,
                 dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, deep_supervision, num_classes,
                 num_classes_cls=3, **kwargs):
        
        # Map nnU-Net parameter names to ResidualEncoderUNet parameter names
        # n_conv_per_stage -> n_blocks_per_stage
        n_blocks_per_stage = n_conv_per_stage
        
        # Initialize the parent ResidualEncoderUNet
        super().__init__(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_blocks_per_stage=n_blocks_per_stage,  # Mapped from n_conv_per_stage
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
        
        # Get encoder features from the bottleneck (last encoder stage)
        encoder_features = features_per_stage[-1]
        
        # Enhanced classification head with proper normalization
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.global_max_pool = nn.AdaptiveMaxPool3d(1)
        
        # Classification head with batch normalization and proper regularization
        self.classification_head = nn.Sequential(
            # Feature normalization layer
            nn.BatchNorm1d(encoder_features * 2),  # *2 for avg + max pooling
            nn.Dropout(0.3),
            nn.Linear(encoder_features * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes_cls)
        )
        
        # Store for classification output access
        self.last_classification_output = None
        self._bottleneck_features = None
        
        # Register a more reliable hook on the encoder
        self._register_encoder_hook()
        
        print(f"Multi-task ResidualEncoderUNet initialized with {encoder_features} encoder features")
        print("Using improved hook-based feature extraction for robust classification")
    
    def _register_encoder_hook(self):
        """Register a robust hook on the encoder to capture bottleneck features."""
        def encoder_hook(module, input, output):
            # Store the bottleneck features for classification
            self._bottleneck_features = output
        
        # Register hook on the last encoder stage
        if hasattr(self.encoder, 'stages') and len(self.encoder.stages) > 0:
            last_encoder_stage = self.encoder.stages[-1]
            self._hook_handle = last_encoder_stage.register_forward_hook(encoder_hook)
            print(f"Registered improved hook on encoder bottleneck: {type(last_encoder_stage)}")
        else:
            print("Warning: Could not find encoder stages for hook registration")
            self._hook_handle = None
    
    def forward(self, x):
        """Forward pass using parent's method with improved feature capture."""
        # Clear previous features
        self._bottleneck_features = None
        
        # Use parent's forward method for segmentation (maintains proper channel flow)
        seg_output = super().forward(x)
        
        # Process classification if we captured features
        if self._bottleneck_features is not None:
            # Apply proper feature normalization and pooling
            avg_pooled = self.global_avg_pool(self._bottleneck_features)
            max_pooled = self.global_max_pool(self._bottleneck_features)
            
            # Flatten and concatenate
            avg_pooled = avg_pooled.view(avg_pooled.size(0), -1)
            max_pooled = max_pooled.view(max_pooled.size(0), -1)
            
            # L2 normalize features to prevent scale issues
            avg_pooled = nn.functional.normalize(avg_pooled, p=2, dim=1)
            max_pooled = nn.functional.normalize(max_pooled, p=2, dim=1)
            
            pooled_features = torch.cat([avg_pooled, max_pooled], dim=1)
            
            # Classification prediction
            cls_output = self.classification_head(pooled_features)
            
            # Store classification output for trainer access
            self.last_classification_output = cls_output
        else:
            # Fallback: create dummy classification output
            batch_size = x.size(0)
            self.last_classification_output = torch.zeros(batch_size, self.num_classes_cls, device=x.device)
            print("Warning: No bottleneck features captured, using dummy classification output")
        
        # Return segmentation output in the format nnU-Net expects
        return seg_output
    
    def forward_segmentation_only(self, x):
        """Forward pass for segmentation only using parent's method."""
        return super().forward(x)
    
    def forward_classification_only(self, x):
        """Forward pass for classification only."""
        # Run encoder only to get features
        self._bottleneck_features = None
        
        # Run through encoder stages to capture bottleneck features
        current = x
        for encoder_stage in self.encoder.stages:
            current = encoder_stage(current)
        
        # Use the final encoder output as bottleneck features
        self._bottleneck_features = current
        
        # Process features for classification
        if self._bottleneck_features is not None:
            avg_pooled = self.global_avg_pool(self._bottleneck_features)
            max_pooled = self.global_max_pool(self._bottleneck_features)
            
            avg_pooled = avg_pooled.view(avg_pooled.size(0), -1)
            max_pooled = max_pooled.view(max_pooled.size(0), -1)
            
            # L2 normalize features
            avg_pooled = nn.functional.normalize(avg_pooled, p=2, dim=1)
            max_pooled = nn.functional.normalize(max_pooled, p=2, dim=1)
            
            pooled_features = torch.cat([avg_pooled, max_pooled], dim=1)
            return self.classification_head(pooled_features)
        else:
            raise ValueError("No bottleneck features captured for classification")
    
    def __del__(self):
        """Clean up hook when object is destroyed."""
        if hasattr(self, '_hook_handle') and self._hook_handle is not None:
            self._hook_handle.remove() 