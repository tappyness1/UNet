from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import os 
import torch
from torchvision import transforms 
from torch.utils.data import DataLoader


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.images = os.listdir(images_dir)
        self.masks = os.listdir(mask_dir)
        self.images.sort()
        self.masks.sort()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        transform = transforms.Compose([transforms.PILToTensor(),
                                        transforms.Resize((512, 512))])
        mask = transform(Image.open(os.path.join(self.mask_dir, self.masks[idx]))).to(torch.float32)
        img = transform(Image.open(os.path.join(self.images_dir, self.images[idx]))).to(torch.float32)
        img /= 255
        mask /= 255
        return [img, mask]

if __name__ == "__main__":
    images_dir = "../data/carvana/train"
    masks_dir = "../data/carvana/train_masks"
    dataset = BasicDataset(images_dir, masks_dir)
    train_dataloader = DataLoader(dataset, batch_size=6, shuffle = True)
    img, mask = next(iter(train_dataloader))
    print (mask.unique())
