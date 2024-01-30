from typing import Union, List, Optional, Callable

import torchvision.datasets
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


class CelebA(Dataset):
    """Extend torchvision.datasets.CelebA with three pre-defined transforms.

    The pre-defined transforms are:
      - 'stylegan-like' (default): Crop [cy-64 : cy+64, cx-64 : cx+64] where cx=89 and cy=121
      - 'resize': Resize the image directly
      - 'crop140': Crop 140x140 from the center of the image
    All of the above transforms will be followed by resizing and random horizontal flipping.

    """

    def __init__(
            self,
            root: str,
            img_size: int,
            split: str = 'train',
            target_type: Union[List[str], str] = "attr",
            transform_type: str = 'default',
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ):
        if split not in ['train', 'valid', 'test', 'all']:
            raise ValueError(f'Invalid split: {split}')

        self.img_size = img_size
        self.split = split
        self.transform_type = transform_type
        if transform is None:
            transform = self.get_transform()

        self.celeba = torchvision.datasets.CelebA(
            root=root,
            split=split,
            target_type=target_type,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

    def __len__(self):
        return len(self.celeba)

    def __getitem__(self, item):
        X, y = self.celeba[item]
        return X, y

    def get_transform(self):
        flip_p = 0.5 if self.split in ['train', 'all'] else 0.0
        if self.transform_type in ['default', 'stylegan-like']:
            # https://github.com/NVlabs/stylegan/blob/master/dataset_tool.py#L484-L499
            cx, cy = 89, 121
            transform = T.Compose([
                T.Lambda(lambda x: TF.crop(x, top=cy-64, left=cx-64, height=128, width=128)),
                T.Resize((self.img_size, self.img_size), antialias=True),
                T.RandomHorizontalFlip(flip_p),
                T.ToTensor(),
                T.Normalize([0.5] * 3, [0.5] * 3),
            ])
        elif self.transform_type == 'resize':
            transform = T.Compose([
                T.Resize((self.img_size, self.img_size), antialias=True),
                T.RandomHorizontalFlip(flip_p),
                T.ToTensor(),
                T.Normalize([0.5] * 3, [0.5] * 3),
            ])
        elif self.transform_type == 'crop140':
            transform = T.Compose([
                T.CenterCrop((140, 140)),
                T.Resize((self.img_size, self.img_size), antialias=True),
                T.RandomHorizontalFlip(flip_p),
                T.ToTensor(),
                T.Normalize([0.5] * 3, [0.5] * 3),
            ])
        elif self.transform_type == 'none':
            transform = None
        else:
            raise ValueError(f'Invalid transform_type: {self.transform_type}')
        return transform
