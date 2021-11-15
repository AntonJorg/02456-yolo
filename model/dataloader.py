import os
import io
import re
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class HELMETDataSet(Dataset):
    def __init__(self, root_dir):
        # working directories
        self.root_dir = root_dir
        self.video_dir = os.path.join(root_dir, "image")
        self.annotation_dir = os.path.join(root_dir, "annotation")

        # image paths and their corresponding video + frame ID
        self.images_paths = []
        self.images_vid_names = []
        self.images_id = []
        self.images_annotations = []
        prev_root = ""
        for root, dirs, files in os.walk(self.video_dir):
            if not dirs:
                if prev_root != root:
                    vid_name = os.path.split(root)[-1]
                    csv = pd.read_csv(os.path.join(self.annotation_dir, vid_name + ".csv"))
                    csv = np.array(csv)
                for file in files:
                    if file.endswith(".jpg"):  # avoid .DSStore
                        self.images_paths.append(os.path.join(root, file))
                        vid_name = os.path.split(root)[-1]
                        self.images_vid_names.append(vid_name)
                        im_id = int(file.split(".")[0])
                        self.images_id.append(im_id)
                        self.images_annotations.append(csv[csv[:, 1] == im_id])
                prev_root = root

        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        # input
        x = Image.open(self.images_paths[idx])

        # target
        y = self.images_annotations[idx]
        y = {i: row for i, row in enumerate(y)}

        return self.transform(x), y


class HELMETDataLoader(DataLoader):
    def __init__(self, root_dir, shuffle=True, batch_size=4):
        dataset = HELMETDataSet(root_dir)
        super().__init__(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=self.custom_collate_fn)

    def custom_collate_fn(self, batch):
        return tuple(zip(*batch))


def dict_from_bounding_box(bb):
    d = {"track_id": bb[0], "frame_id": bb[1], "dim": bb[2:6], "class": bb[6]}
    return d


def pos_encoding_from_label(label):
    pos_encoding = torch.zeros(10)

    for i, sub in enumerate(label.split("P")):
        if sub[0] == "D":
            ind = 0
        else:
            ind = int(sub[0]) + 1

        pos_encoding[ind] = 1
        if sub[1:] == "Helmet":
            pos_encoding[ind + 5] = 1

    return pos_encoding


if __name__ == "__main__":
    label = "DHelmetP0NoHelmetP1HelmetP2NoHelmet"
    print(pos_encoding_from_label(label))
