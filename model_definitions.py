# backend/model_definitions.py

import torch.nn as nn
from torchvision import models

class MultiOutputModel(nn.Module):
    """The model for classifying cattle breed and type."""
    def __init__(self, num_breeds, num_types):
        super().__init__()
        # Use the newer 'weights' parameter instead of 'pretrained'
        self.base_model = models.mobilenet_v2(weights='IMAGENET1K_V1')

        # Freeze early layers
        for param in self.base_model.features[:-4].parameters():
            param.requires_grad = False

        # Replace the final classifier
        num_ftrs = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_ftrs, 512),
            nn.ReLU()
        )

        # Create separate heads for each prediction task
        self.breed_head = nn.Linear(512, num_breeds)
        self.type_head = nn.Linear(512, num_types)

    def forward(self, x):
        x = self.base_model(x)
        # Return logits from each head
        return self.breed_head(x), self.type_head(x)

def create_health_model(num_classes=2):
    """The model for classifying cattle health."""
    model = models.mobilenet_v2(weights='IMAGENET1K_V1')

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final classifier layer
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return model