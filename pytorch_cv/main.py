import albumentations as alb
from albumentations.pytorch import ToTensorV2
import numpy as np

class TrainTransforms:
    def __init__(self, pMean, pStd):
        self.train_transform = alb.Compose([
            alb.Resize(36, 36, 3),
            alb.RandomResizedCrop(32, 32),
            #alb.HorizontalFlip(),
            #alb.ShiftScaleRotate(),
            alb.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16, fill_value=[i * 2555 for i in pMean], mask_fill_value=None), # type: ignore
            alb.Normalize(mean=pMean, std=pStd),
            ToTensorV2(),
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.train_transform(image=img)['image']
        return img


class TestTransforms:
    def __init__(self, pMean, pStd):
        self.test_transform = alb.Compose([
            alb.Resize(32, 32, 3),
            alb.Normalize(mean=pMean, std=pStd),
            ToTensorV2(),
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.test_transform(image=img)['image']
        return img