import os
import cv2
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_transforms(high_res: int = 416):
    low_res = high_res // 4

    both_transforms = A.Compose(
        [
            A.RandomCrop(width=high_res, height=high_res),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ]
    )

    highres_transform = A.Compose(
        [
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
            ToTensorV2(),
        ]
    )

    lowres_transform = A.Compose(
        [
            A.Resize(width=low_res, height=low_res, interpolation=Image.BICUBIC),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
            ToTensorV2(),
        ]
    )

    test_transform = A.Compose(
        [
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
            ToTensorV2(),
        ]
    )

    return both_transforms, lowres_transform, highres_transform, test_transform


class MyImageFolder(Dataset):
    def __init__(self, root_dir, high_res=416):
        super().__init__()
        self.data = []
        self.root_dir = root_dir
        self.class_names = sorted(os.listdir(root_dir))
        self.both_transforms, self.lowres_transform, self.highres_transform, _ = build_transforms(high_res)

        for index, name in enumerate(self.class_names):
            class_dir = os.path.join(root_dir, name)
            if not os.path.isdir(class_dir):
                continue
            files = os.listdir(class_dir)
            self.data += list(zip(files, [index] * len(files)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_file, label = self.data[index]
        root_and_dir = os.path.join(self.root_dir, self.class_names[label])

        image = cv2.imread(os.path.join(root_and_dir, img_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        both_transform = self.both_transforms(image=image)["image"]
        low_res = self.lowres_transform(image=both_transform)["image"]
        high_res = self.highres_transform(image=both_transform)["image"]
        return low_res, high_res