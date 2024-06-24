import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import KFold
from tqdm import tqdm

from dataset import SatelliteDataset
from model import UNet

# Hyperparameters
NUM_EPOCHS = 10
LEARNING_RATE = 0.01
LEARNING_RATE_DECAY = 1e-4
BATCH_SIZE = 4
NUM_FOLDS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Record directory
attempt = 'f_1'
model_dir = f'./model/{attempt}'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

dataset = SatelliteDataset(
    image_dir="./src/data/buildings/train_images/imgs/",
    mask_dir="./src/data/buildings/train_masks/masks/"
)

kfold = KFold(n_splits=NUM_FOLDS, shuffle=True)
criterion = nn.BCEWithLogitsLoss()

print('========================== Training Started ==========================')

for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
    print(f'Fold {fold+1}/{NUM_FOLDS}')

    train_subsampler = Subset(dataset, train_idx)
    val_subsampler = Subset(dataset, val_idx)
    train_loader = DataLoader(train_subsampler, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subsampler, batch_size=BATCH_SIZE, shuffle=False)

    model = UNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        train_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}, Training', leave=False)

        for images, masks in train_progress:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE).unsqueeze(1).float()

            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            train_progress.set_postfix(loss=(running_loss / len(train_subsampler)))

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Fold {fold+1} Epoch {epoch+1}/{NUM_EPOCHS}, Training Loss: {epoch_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_progress = tqdm(val_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}, Validation', leave=False)
        with torch.no_grad():
            for images, masks in val_progress:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE).unsqueeze(1).float()

                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)

                val_loss += loss.item() * images.size(0)
                val_progress.set_postfix(val_loss=(val_loss / len(val_subsampler)))

        val_loss /= len(val_loader.dataset)
        print(f"Fold {fold+1} Epoch {epoch+1}/{NUM_EPOCHS}, Validation Loss: {val_loss:.4f}")
        
    # Save
    torch.save(model.state_dict(), f"{model_dir}/unet_fold_{fold+1}.pth")
