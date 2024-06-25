import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
from parser import args
from PIL import Image
import numpy as np


def make_dataset(root_path):
    images = []

    photo_root = os.path.join(root_path, '256x256', 'photo', 'tx_000000000000')
    sketch_root = os.path.join(root_path, '256x256', 'sketch', 'tx_000000000000')  # 如需更改所用数据类型，直接更改tx之后的数字即可
    classes = os.listdir(sketch_root)
    for c in classes:
        sketches = os.listdir(os.path.join(sketch_root, c))
        for sketch_name in sketches:
            suffix_idx = sketch_name.index('-')
            img_name = sketch_name[:suffix_idx] + '.jpg'
            images.append((os.path.join(photo_root, c, img_name), os.path.join(sketch_root, c, sketch_name)))
    return images


class ImageDataset(Dataset):
    def __init__(self, args):
        self.image_transform = transforms.Compose([
            # transforms.Resize(224),
            # transforms.CenterCrop(224),  # 如需使用224x224大小输入，取消这两行的注释即可
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.sketch_transform = transforms.Compose([
            # transforms.Resize(224),
            # transforms.CenterCrop(224),  # 同上
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])
        self.data_path = args.data_path
        self.image_paths = make_dataset(self.data_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index][0]
        sketch_path = self.image_paths[index][1]

        image = Image.open(image_path).convert('RGB')
        image = self.image_transform(image)
        sketch = Image.open(sketch_path).convert('1')
        sketch = self.sketch_transform(sketch)

        return image, sketch


# an example:
if __name__ == '__main__':
    dataset = ImageDataset(args)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)
    for i, batch in enumerate(dataloader):
        img_batch = batch[0]
        sketch_batch = batch[1]
        print(img_batch.shape)  # 16, 3, 256, 256
        print(sketch_batch.shape)  # 16, 1, 256, 256
#  已完成，直接使用pytorch的Dataloader即可，dataloader输出为[img batch, sketch batch]，每个batch shape为(bsz, channel, height, width)