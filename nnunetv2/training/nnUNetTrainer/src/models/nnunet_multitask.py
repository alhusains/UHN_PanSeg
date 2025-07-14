import torch
import torch.nn as nn
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from nnunetv2.utilities.network_initialization import InitWeights_He

#classification head
class ClassificationHead(nn.Module):
    def __init__(self, encoder_channels, num_classes=3, hidden_dim=256, dropout_p=0.3):
        super().__init__()
        
        #validate input
        if len(encoder_channels) < 3:
            raise ValueError("Encoder must have at least 3 feature maps")
        
        #take the last 3 feature maps
        selected_channels = encoder_channels[-3:]
        
        #Feature compression (adapts to actual channel dims)
        self.feature_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(ch, 64, kernel_size=1, bias=False),
                nn.BatchNorm3d(64),
                nn.GELU(),
                nn.AdaptiveAvgPool3d((4, 4, 4))
            ) for ch in selected_channels
        ])
        
        #enhanced fusion
        self.fusion = nn.Sequential(
            nn.Conv3d(64 * 3, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool3d(1)
        )
        
        #classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        #initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, encoder_features):
        #input validation
        if len(encoder_features) < 3:
            raise ValueError(f"Expected ≥3 feature maps, got {len(encoder_features)}")
        
        #process last 3 features
        adapted = []
        for i, adapter in enumerate(self.feature_adapters):
            feat = encoder_features[-(3 - i)]  # Get features from last to -3
            adapted.append(adapter(feat))
        
        #concatenate and fuse
        x = torch.cat(adapted, dim=1)
        x = self.fusion(x).flatten(1)
        
        return self.classifier(x)

class nnUNetv2MultiTask(ResidualEncoderUNet):    
    def __init__(self, input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                 n_conv_per_stage, n_conv_per_stage_decoder, conv_bias, norm_op, norm_op_kwargs,
                 dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, deep_supervision, num_classes,
                 num_classes_cls=3, **kwargs):
        
        # Map nnU-Net parameter names to ResidualEncoderUNet parameter names
        n_blocks_per_stage = n_conv_per_stage
        
        #initialize the parent ResidualEncoderUNet
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
        
        #use classification head
        encoder_output_channels = self.encoder.output_channels
        self.ClassificationHead = ClassificationHead(
            encoder_output_channels,
            num_classes=num_classes_cls
        )
        
        #initialize classification head
        self.ClassificationHead.apply(InitWeights_He(1e-2))
        
        print(f"Multi-task ResidualEncoderUNet initialized with encoder channels: {encoder_output_channels}")
    
    def forward(self, x):
        #get segmentation output from parent
        seg_output = super().forward(x)
        
        #get encoder features directly
        enc_features = self.encoder(x)
        
        #get classification output using all encoder features
        class_logits = self.ClassificationHead(enc_features)
        
        #store for trainer access
        self.last_classification_output = class_logits
        
        return seg_output 