import torch
import imageio.v2 as imageio
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, image_root):
        self.tranform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.image_root = image_root
        self.image_paths = sorted(make_dataset(self.image_root))

        self.len = len(self.image_paths)
#  待修改，预计dataloader输出为(sketch batch, img batch)，每个batch shape为(bsz, channel, height, width)