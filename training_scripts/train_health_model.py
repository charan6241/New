# =============================================================================
# SCRIPT: train_health_model.py
# PURPOSE: Train a classifier for healthy vs. unhealthy appearance.
# =============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
import time

# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================
if __name__ == '__main__':
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"Is CUDA available? {torch.cuda.is_available()}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- CONFIGURATION ---
    # ‼️ IMPORTANT: UPDATE THIS PATH to your new health dataset folder
    DATA_PATH = r"D:\sih_2025\cattle_health_class\train" 
    
    NUM_EPOCHS = 10 # This is a simpler problem, so 10 epochs is a good start
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001

    # --- DATA PREPARATION ---
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Using the standard ImageFolder, which is perfect for this simple structure
    full_dataset = datasets.ImageFolder(root=DATA_PATH, transform=data_transforms)
    dataloader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # Get class names (e.g., 'Healthy_Appearance', 'Potential_Concern')
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    # --- MODEL DEFINITION ---
    model = models.mobilenet_v2(weights='IMAGENET1K_V1')
    
    # Freeze base layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final layer for our 2-class problem
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    model.to(device)

    # --- TRAINING SETUP ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier[1].parameters(), lr=LEARNING_RATE) # Only train the new layer

    # --- TRAINING LOOP ---
    print("Starting health model training...")
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(full_dataset)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} completed. Average Loss: {epoch_loss:.4f}")

    end_time = time.time()
    print(f"Training finished in {(end_time - start_time)/60:.2f} minutes.")

    # --- SAVE THE MODEL ---
    # Save to a DIFFERENT file to avoid overwriting your breed model
    model_data = {
        "model_state": model.state_dict(),
        "class_names": class_names
    }
    torch.save(model_data, "health_classifier_model.pth")
    print("Health model saved to health_classifier_model.pth")