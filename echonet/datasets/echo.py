import collections
import glob
import os
import re
import numpy as np
import pandas as pd
import skimage.draw
import torch
import torch.utils.data
import torchvision
from torchvision import transforms

class Echo(torch.utils.data.Dataset):
    """EchoNet-Dynamic dataset."""

    def __init__(self, root=None, split="train", target_type="EF", mean=0., std=1., length=32, period=1, max_length=250, pad=None):
        if isinstance(target_type, str):
            target_type = [target_type]
        self.target_type = target_type
        assert split in ["train", "val", "test", "all", "external_test"]
        assert all(t in ["EF", "LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace", "Filename", "Size", "FrameHeight", "FrameWidth", "FPS"] for t in self.target_type)

        self.root = root
        self.split = split.upper()
        self.mean = mean
        self.std = std
        self.length = length
        self.max_length = max_length
        self.period = period
        self.pad = pad
        self.fnames = []
        self.outcomes = []

        # Load csv file
        if self.root is not None:
            data_csv = os.path.join(self.root, "FileList.csv")
            df = pd.read_csv(data_csv)
            if self.split != "all":
                df = df[df["Split"] == self.split]
            self.fnames = df["FileName"].tolist()
            self.outcomes = df[self.target_type].values

            if len(self.fnames) == 0:
                raise ValueError("No videos found in {} split".format(self.split))

        # Define transformations
        if split == "train":
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[self.mean], std=[self.std])  # Normalize with dataset mean/std
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[self.mean], std=[self.std])
            ])

    def __getitem__(self, index):
        # Find filename of video
        video_path = os.path.join(self.root, "Videos", self.fnames[index])

        # Load video into np.array
        video = torchvision.io.read_video(video_path, pts_unit="sec")[0].numpy()

        # Gather targets
        target = self.outcomes[index]
        if len(self.target_type) == 1:
            target = target[0]

        # Trim video
        if self.length is not None:
            if len(video) >= self.length:
                start = np.random.randint(0, len(video) - self.length + 1)
                video = video[start:start + self.length]
            else:
                video = np.pad(video, [(0, self.length - len(video)), (0, 0), (0, 0), (0, 0)], mode="edge")

        # Select random frame for LargeFrame/SmallFrame
        if any(t in self.target_type for t in ["LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace"]):
            frame_idx = np.random.randint(0, len(video))
            frame = video[frame_idx]
            if "LargeFrame" in self.target_type or "SmallFrame" in self.target_type:
                target = [target, frame] if isinstance(target, float) else [*target, frame]
            if "LargeTrace" in self.target_type or "SmallTrace" in self.target_type:
                trace_path = os.path.join(self.root, "Trace", self.fnames[index].replace(".avi", ".csv"))
                trace = pd.read_csv(trace_path)
                target = [target, trace] if isinstance(target, float) else [*target, trace]

        # Apply transforms
        if self.transform is not None:
            video = [self.transform(torch.from_numpy(frame).permute(2, 0, 1)) for frame in video]
            video = torch.stack(video)

        return video, target

    def __len__(self):
        return len(self.fnames)
