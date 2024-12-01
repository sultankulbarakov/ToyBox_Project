# src/model.py
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional

class ViewEncoder(nn.Module):
    # Encoder network for processing individual views
    def __init__(self, pretrained: bool = True, feature_dim: int = 2048):
        super(ViewEncoder, self).__init__()
        # Using ResNet50 as the backbone
        resnet = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        # Removing the final classification layer
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        # Adding feature projection layer
        self.projection = nn.Sequential(
            nn.Linear(2048, feature_dim),
            nn.ReLU(),
            nn.BatchNorm1d(feature_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, channels, height, width)
        features = self.encoder(x)
        # Flatten features
        features = features.view(features.size(0), -1)
        # Project features
        return self.projection(features)

class LateFusionModel(nn.Module):
    # Multi-view object classification model using late fusion strategy
    def __init__(
        self,
        num_classes: int,
        num_views: int = 3,
        feature_dim: int = 2048,
        dropout_rate: float = 0.5,
        fusion_method: str = 'concat'
    ):
        super(LateFusionModel, self).__init__()
        self.num_views = num_views
        self.fusion_method = fusion_method
        # Viewing encoder (shared weights across views)
        self.view_encoder = ViewEncoder(pretrained=True, feature_dim=feature_dim)
        # Calculating fusion output dimension
        fusion_dim = feature_dim * num_views if fusion_method == 'concat' else feature_dim
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(fusion_dim // 2),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim // 2, num_classes)
        )
        
    def fuse_features(self, view_features: torch.Tensor) -> torch.Tensor:
        """Fuse features from multiple views."""
        # view_features shape: (batch_size, num_views, feature_dim)
        if self.fusion_method == 'concat':
            # Concatenating features from all views
            return view_features.view(view_features.size(0), -1)
        elif self.fusion_method == 'mean':
            # Average features across views
            return torch.mean(view_features, dim=1)
        elif self.fusion_method == 'max':
            # Max pooling across views
            return torch.max(view_features, dim=1)[0]
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion_method}")
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, num_views = x.size(0), x.size(1)
        # Reshaping input to process all views
        x = x.view(-1, *x.shape[2:])
        # Extracting features for each view
        view_features = self.view_encoder(x)
        # Reshaping features back to separate views
        view_features = view_features.view(batch_size, num_views, -1)
        fused_features = self.fuse_features(view_features)
        # Classification
        logits = self.classifier(fused_features)
        if return_features:
            return logits, fused_features
        return logits, None

def create_model(
    num_classes: int,
    num_views: int = 3,
    feature_dim: int = 2048,
    dropout_rate: float = 0.5,
    fusion_method: str = 'concat'
) -> LateFusionModel:
    # Creating late fusion model instance
    model = LateFusionModel(
        num_classes=num_classes,
        num_views=num_views,
        feature_dim=feature_dim,
        dropout_rate=dropout_rate,
        fusion_method=fusion_method
    )
    return model

if __name__ == "__main__":
    # Testing model
    model = create_model(num_classes=100, num_views=3)
    # Creating dummy input
    dummy_input = torch.randn(2, 3, 3, 224, 224) 
    # Forwarding pass
    logits, _ = model(dummy_input)
    print(f"Output shape: {logits.shape}") 