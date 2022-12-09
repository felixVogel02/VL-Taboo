
import os

import torch
from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset


class LabeledDatasetNew(Dataset):
    """Dataset for image-wise attribute loading of CUB."""

    def __init__(self, image_attr, label_idx, idx_attr, all_imgs_encoded, root_dir="/data/felix/AWA2/Animals_with_Attributes2/JPEGImages/"):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.image_attr = image_attr
        self.label_idx = label_idx
        self.idx_attr = idx_attr
        self.all_imgs_encoded = all_imgs_encoded

        self.root_dir = root_dir
        all_folders = os.listdir(root_dir)
        all_imgs = []
        for folder in all_folders:
            all_imgs.extend([os.path.join(folder, i) for i in os.listdir(os.path.join(root_dir, folder))])
        self.all_imgs = natsorted(all_imgs)

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, id):
        if torch.is_tensor(id):
            id = id.tolist()

        img_name = self.all_imgs[id]
        img_loc = os.path.join(self.root_dir, img_name)

        image = self.all_imgs_encoded[id]

        label_name = img_loc.split("/")[-2]  # .replace("+", " ")
        class_id = self.label_idx[label_name]
        indices = self.image_attr[id]
        attributes = {i.item(): self.idx_attr[i.item()] for i in indices}
        # print("Indices dataloader: ", indices)

        sample = {'image': image, "img_attr_ids": indices, "img_attrs": attributes,
                  "label_name": label_name, "class_id": class_id, "image_name": img_name, "img_id": id}

        return sample
