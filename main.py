from dataset import SatelliteDataset
from model import UNet
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import KFold

# Hyperparameters
NUM_EPOCHS = 10
LEARNING_RATE = 0.01
BATCH_SIZE = 4
NUM_FOLDS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# record
attempt = 'f_1'
if not os.path.exists(f'./model/{attempt}'):
    os.mkdir(f'./model/{attempt}')

dataset = SatelliteDataset(
    image_dir = "./src/data/buildings/train_images/imgs/", 
    mask_dir = "./src/data/buildings/train_masks/masks/"
    )
kfold = KFold(n_splits=NUM_FOLDS, shuffle=True)

model = UNet().cuda()
optimizer = torch.optim.Adam(model.parameters())
scaler = GradScaler()
criterion = nn.BCELoss()

print('========================Traing started========================')

for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
    print(f'Fold {fold+1}/{NUM_FOLDS}')
    
    # Subset random samplers & Data loaders
    train_subsampler = Subset(dataset, train_idx)
    val_subsampler = Subset(dataset, val_idx)
    train_loader = DataLoader(train_subsampler, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subsampler, batch_size=BATCH_SIZE, shuffle=False)

    # Model, optimizer, loss function
    model = UNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            # Scale the loss and backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Training Loss: {epoch_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                
                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Validation Loss: {val_loss:.4f}")
        
    torch.save(model.state_dict(), f"./model/{attempt}/unet_fold_{fold+1}.pth")