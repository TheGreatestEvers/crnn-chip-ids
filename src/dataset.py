# In file: src/dataset.py

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
# The 'transform' argument is now accepted. 'torchvision.transforms' will be used here.

class CustomOCRDataset(Dataset):
    """
    Custom Dataset for Optical Character Recognition.
    Reads image paths and labels from a pandas DataFrame and applies optional transforms.
    """
    def __init__(self, df: pd.DataFrame, root_dir: str, chars: str,
                 img_height=32, img_width=100, transform=None): # <-- ADDED transform=None
        
        self.df = df
        self.root_dir = root_dir
        self.CHAR2LABEL = {char: i + 1 for i, char in enumerate(chars)}
        self.LABEL2CHAR = {label: char for char, label in self.CHAR2LABEL.items()}
        
        self.paths = self.df['file_name'].values
        self.texts = self.df['text'].values
        
        self.img_height = img_height
        self.img_width = img_width
        self.transform = transform # <-- STORE THE TRANSFORM

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_path = os.path.join(self.root_dir, self.paths[index])
        text = self.texts[index]

        try:
            image = Image.open(image_path).convert('L')
        except IOError:
            print(f'Corrupted image for {image_path}')
            return self[(index + 1) % len(self)]

        # <-- APPLY TRANSFORMS HERE (if they exist) -->
        # The transforms are applied to the PIL image before resizing and normalization.
        if self.transform:
            image = self.transform(image)

        # Standard processing (resizing, normalization) happens after augmentation
        image = image.resize((self.img_width, self.img_height), resample=Image.BILINEAR)
        image = np.array(image)
        
        image = image.reshape((1, self.img_height, self.img_width))
        image = (image / 127.5) - 1.0 # Normalize to [-1.0, 1.0]
        
        image = torch.FloatTensor(image)

        target = [self.CHAR2LABEL[c] for c in text]
        target_length = [len(target)]
        target = torch.LongTensor(target)
        target_length = torch.LongTensor(target_length)

        return image, target, target_length

# The collate function remains the same
def ocr_collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths