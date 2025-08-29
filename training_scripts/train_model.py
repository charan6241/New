# =============================================================================
# 1. SETUP - Importing necessary libraries
# =============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os
import time

# =============================================================================
# CLASS AND FUNCTION DEFINITIONS
# These are defined at the top level so they can be accessed by all processes
# =============================================================================

# Custom Dataset class to handle two labels
class BovineDataset(Dataset):
    def __init__(self, root_dir, transform=None, breed_map=None, type_map=None):
        self.root_dir = root_dir
        self.transform = transform
        self.breed_to_idx = breed_map
        self.type_to_idx = type_map
        self.image_paths = []
        self.breed_labels = []
        self.type_labels = []
        
        class_names = sorted([name for name in os.listdir(root_dir) if '_' in name and os.path.isdir(os.path.join(root_dir, name))])
        
        for class_name in class_names:
            breed, animal_type = class_name.rsplit('_', 1)
            class_dir = os.path.join(root_dir, class_name)
            
            for img_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, img_name))
                self.breed_labels.append(self.breed_to_idx[breed])
                self.type_labels.append(self.type_to_idx[animal_type])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        breed_label = self.breed_labels[idx]
        type_label = self.type_labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, breed_label, type_label

# Multi-output model definition
class MultiOutputModel(nn.Module):
    def __init__(self, num_breeds, num_types):
        super().__init__()
        self.base_model = models.mobilenet_v2(weights='IMAGENET1K_V1')
        
        for param in self.base_model.features[:-4].parameters():
            param.requires_grad = False
            
        num_ftrs = self.base_model.classifier[1].in_features
        
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
        )
        self.breed_head = nn.Linear(512, num_breeds)
        self.type_head = nn.Linear(512, num_types)

    def forward(self, x):
        x = self.base_model(x)
        breed_output = self.breed_head(x)
        type_output = self.type_head(x)
        return breed_output, type_output

# =============================================================================
# MAIN EXECUTION BLOCK
# This code will only run when the script is executed directly
# =============================================================================
if __name__ == '__main__':
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"Is CUDA available? {torch.cuda.is_available()}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- CONFIGURATION ---
    DATA_PATH = r"D:\sih_2025\SIH_Breeds_Dataset\archive_dataset\Indian_bovine_breeds" 
    NUM_EPOCHS = 15
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001

    # --- DATA PREPARATION ---
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    all_folders = [d.name for d in os.scandir(DATA_PATH) if d.is_dir()]
    class_names = sorted([name for name in all_folders if '_' in name])
    print(f"Found {len(class_names)} valid class folders.")

    unique_breeds = sorted(list(set([n.rsplit('_', 1)[0] for n in class_names])))
    unique_types = sorted(list(set([n.rsplit('_', 1)[1] for n in class_names])))

    breed_to_idx = {breed: i for i, breed in enumerate(unique_breeds)}
    type_to_idx = {animal_type: i for i, animal_type in enumerate(unique_types)}
    
    full_dataset = BovineDataset(root_dir=DATA_PATH, transform=data_transforms, breed_map=breed_to_idx, type_map=type_to_idx)
    dataloader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    num_breed_classes = len(breed_to_idx)
    num_type_classes = len(type_to_idx)
    print(f"Found {num_breed_classes} breed classes and {num_type_classes} type classes for training.")

    # --- MODEL DEFINITION ---
    model = MultiOutputModel(num_breeds=num_breed_classes, num_types=num_type_classes)
    model.to(device)

    # --- TRAINING SETUP ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- TRAINING LOOP ---
    print("Starting training...")
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, (images, breed_labels, type_labels) in enumerate(dataloader):
            images = images.to(device)
            breed_labels = breed_labels.to(device)
            type_labels = type_labels.to(device)
            
            optimizer.zero_grad()
            
            breed_outputs, type_outputs = model(images)
            loss_breed = criterion(breed_outputs, breed_labels)
            loss_type = criterion(type_outputs, type_labels)
            total_loss = loss_breed + loss_type
            
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item() * images.size(0)
            
            if (i + 1) % 50 == 0:
                print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(dataloader)}], Loss: {total_loss.item():.4f}')

        epoch_loss = running_loss / len(full_dataset)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} completed. Average Loss: {epoch_loss:.4f}")

    end_time = time.time()
    print(f"Training finished in {(end_time - start_time)/60:.2f} minutes.")

    # --- SAVE THE MODEL ---
    model_data = {
        "model_state": model.state_dict(),
        "breed_to_idx": breed_to_idx,
        "type_to_idx": type_to_idx
    }
    torch.save(model_data, "bovine_classifier_model.pth")
    print("Model saved to bovine_classifier_model.pth")