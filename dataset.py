import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class SatelliteDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.crop_size = 480

    def __len__(self):
        return len(self.images) * 16
    
    def __getitem__(self, idx):
        file_idx = idx // 16
        crop_idx = idx % 16 
        x = (crop_idx % 4) * self.crop_size
        y = (crop_idx // 4) * self.crop_size

        img_path = os.path.join(self.image_dir, self.images[file_idx])
        mask_path = os.path.join(self.mask_dir, self.images[file_idx])

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = image.astype('float32') / 255
        mask = mask.astype('float32') / 255

        # Crop the image and mask
        image = image[y:y+self.crop_size, x:x+self.crop_size]
        mask = mask[y:y+self.crop_size, x:x+self.crop_size]

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        image = np.transpose(image, (2, 0, 1))

        return image, mask
    

if __name__ == "__main__":
    dataset = SatelliteDataset(
        image_dir = "./src/data/buildings/train_images/imgs/", 
        mask_dir = "./src/data/buildings/train_masks/masks/"
        )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    print(f"Length of dataset: {len(dataset)}")
    for images, masks in dataloader:
        fig, axs = plt.subplots(2, 4, figsize=(15, 7))
        for i in range(4):
            img = np.transpose(images[i].numpy(), (1, 2, 0))
            axs[0, i].imshow(img)
            axs[0, i].set_title('Image')
            axs[1, i].imshow(masks[i].numpy(), cmap='gray')
            axs[1, i].set_title('Mask')
        plt.show()
        break 
