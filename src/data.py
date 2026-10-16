"""Data loading and preprocessing for BreastMNIST (MONAI pipeline)."""
from medmnist import BreastMNIST
from monai.data import ArrayDataset, DataLoader, set_track_meta
from monai.transforms import (Compose, EnsureChannelFirst, ScaleIntensityRange,
                              RandFlip, RandRotate, RandZoom)

set_track_meta(False)  # 2D images without spatial metadata: plain tensors


def get_transforms():
    """Return (train_tf, eval_tf). Augmentation is applied to training only."""
    base = [EnsureChannelFirst(channel_dim="no_channel"),                        # (H, W) → (1, H, W)
            ScaleIntensityRange(a_min=0, a_max=255, b_min=0, b_max=1, clip=True)]  # [0, 255] → [0, 1]
    train_tf = Compose(base + [RandFlip(prob=0.5, spatial_axis=1),            # horizontal flip only
                               RandRotate(range_x=0.26, prob=0.5),            # ±15°
                               RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5)])  # ±10%
    eval_tf = Compose(base)
    return train_tf, eval_tf


def get_loaders(batch_size=32, size=224, num_workers=2):
    """Download BreastMNIST and return (splits, train_loader, val_loader, test_loader).

    Note: original labels are 0 = malignant, 1 = normal/benign.
    """
    splits = {s: BreastMNIST(split=s, download=True, size=size) for s in ["train", "val", "test"]}
    train_tf, eval_tf = get_transforms()

    def make_loader(split, tf, shuffle):
        d = splits[split]
        ds = ArrayDataset(img=d.imgs, img_transform=tf, labels=d.labels.ravel())
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return (splits,
            make_loader("train", train_tf, shuffle=True),
            make_loader("val", eval_tf, shuffle=False),
            make_loader("test", eval_tf, shuffle=False))