import os
import io
import re
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class_dict = {
    "DHelmet": 0,
    "DNoHelmet": 1,
    "DHelmetP0Helmet": 2,
    "DHelmetP0NoHelmet": 3,
    "DNoHelmetP0NoHelmet": 4,
    "DHelmetP1Helmet": 5,
    "DNoHelmetP1Helmet": 6,
    "DHelmetP1NoHelmet": 7,
    "DNoHelmetP1NoHelmet": 8,
    "DHelmetP0HelmetP1Helmet": 9,
    "DHelmetP0NoHelmetP1Helmet": 10,
    "DHelmetP0NoHelmetP1NoHelmet": 11,
    "DNoHelmetP0NoHelmetP1Helmet": 12,
    "DNoHelmetP0NoHelmetP1NoHelmet": 13,
    "DNoHelmetP0HelmetP1NoHelmet": 14,
    "DHelmetP1HelmetP2Helmet": 15,
    "DHelmetP1NoHelmetP2Helmet": 16,
    "DHelmetP1HelmetP2NoHelmet": 17,
    "DHelmetP1NoHelmetP2NoHelmet": 18,
    "DNoHelmetP1HelmetP2Helmet": 19,
    "DNoHelmetP1NoHelmetP2Helmet": 20,
    "DNoHelmetP1NoHelmetP2NoHelmet": 21,
    "DHelmetP0HelmetP1HelmetP2Helmet": 22,
    "DHelmetP0HelmetP1NoHelmetP2Helmet": 23,
    "DHelmetP0HelmetP1NoHelmetP2NoHelmet": 24,
    "DHelmetP0NoHelmetP1HelmetP2Helmet": 25,
    "DHelmetP0NoHelmetP1NoHelmetP2Helmet": 26,
    "DHelmetP0NoHelmetP1NoHelmetP2NoHelmet": 27,
    "DNoHelmetP0NoHelmetP1NoHelmetP2NoHelmet": 28,
    "DNoHelmetP0NoHelmetP1NoHelmetP2Helmet": 29,
    "DHelmetP1NoHelmetP2NoHelmetP3NoHelmet": 30,
    "DHelmetP1NoHelmetP2NoHelmetP3Helmet": 31,
    "DNoHelmetP1NoHelmetP2NoHelmetP3NoHelmet": 32,
    "DNoHelmetP0NoHelmetP1NoHelmetP2NoHelmetP3NoHelmet": 33,
    "DHelmetP0NoHelmetP1NoHelmetP2NoHelmetP3Helmet": 34,
    "DHelmetP0NoHelmetP1NoHelmetP2NoHelmetP3NoHelmet": 35
}


class HELMETDataSet(Dataset):
    def __init__(self, root_dir="", resize=None, split=None, filenames=None):
        # working directories
        self.root_dir = root_dir
        self.video_dir = os.path.join(root_dir, "images")
        self.annotation_dir = os.path.join(root_dir, "annotation")
        self.resize = resize
        self.split = split

        self.images_paths = []
        self.images_vid_names = []
        self.images_id = []
        self.images_annotations = []
        self.included_videos = []

        transform_list = [transforms.ToTensor()]
        if self.resize is not None:
            transform_list = [transforms.Resize(self.resize)] + transform_list

        self.transform = transforms.Compose(transform_list)

        if filenames is not None:
            self.images_only = True
            self.images_paths = filenames
        else:
            self.images_only = False
            self.populate()

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        # input
        x = Image.open(self.images_paths[idx])

        if self.images_only:
            return self.transform(x)

        # target
        target = self.images_annotations[idx]

        d = {i: dict_from_bounding_box(row) for i, row in enumerate(target)}

        y = []
        for k, v in d.items():
            y.append(torch.tensor(np.append(class_dict[v["class"]], v["dim"])).unsqueeze(0))

        if len(y) != 0:
            y = torch.cat(y, 0)
            y *= np.array([1, 1 / 1920, 1 / 1080, 1 / 1920, 1 / 1080])
            y[:, 1:3] += y[:, 3:] / 2
        else:
            y = torch.tensor(y)

        return self.transform(x), y, d

    def populate(self):
        if self.split is None:
            self.included_videos = None
        else:
            assert self.split in ["test", "training", "validation"], "Wrong split type!"
            csv = pd.read_csv("./data_split.csv")#os.path.join(self.root_dir, "data_split.csv"))
            self.included_videos = list(csv["video_id"][csv["Set"] == self.split])

        # image paths and their corresponding video + frame ID

        prev_root = ""
        vid_name = ""
        for root, dirs, files in os.walk(self.video_dir):
            if not dirs:
                if prev_root != root:
                    vid_name = os.path.split(root)[-1]
                    csv = pd.read_csv(os.path.join(self.annotation_dir, vid_name + ".csv"))
                    csv = np.array(csv)
                if self.split is not None and vid_name not in self.included_videos:
                    continue
                for file in files:
                    if file.endswith(".jpg"):  # avoid .DSStore
                        self.images_paths.append(os.path.join(root, file))
                        vid_name = os.path.split(root)[-1]
                        self.images_vid_names.append(vid_name)
                        im_id = int(file.split(".")[0])
                        self.images_id.append(im_id)
                        self.images_annotations.append(csv[csv[:, 1] == im_id])
                prev_root = root


class HELMETDataLoader(DataLoader):
    def __init__(self, root_dir="", batch_size=4, shuffle=True, resize=None, split=None, filenames=None):
        dataset = HELMETDataSet(root_dir, resize=resize, split=split, filenames=filenames)
        if filenames is not None:
            batch_size = len(filenames)
        super().__init__(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=self.custom_collate_fn)

    @staticmethod
    def custom_collate_fn(batch):
        if type(batch[0]) is not tuple:
            imgs = [im.unsqueeze(0) for im in batch]
            imgs = torch.cat(imgs, 0)
            return imgs

        # change such that images are one tensor and not tuple of tensors
        imgs, targets, annotations = tuple(zip(*batch))
        imgs = [im.unsqueeze(0) for im in imgs]
        imgs = torch.cat(imgs, 0)
        targets = [t for t in targets]
        #targets = torch.cat(targets, 0)
        return imgs, targets, annotations


def dict_from_bounding_box(bb):
    d = {"track_id": bb[0], "frame_id": bb[1], "dim": bb[2:6].astype(int), "class": bb[6]}
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
    dataloader = HELMETDataLoader("../data/HELMET_DATASET", resize=True, split="validation")

    batch = next(iter(dataloader))

    imgs, targets, annotations = batch

    print(imgs.shape)
    print(len(targets))
    print(targets)
    print(annotations)
    print(dataloader.dataset.images_annotations[:5])

    label = "DHelmetP0NoHelmetP1HelmetP2NoHelmet"
    print(pos_encoding_from_label(label))
