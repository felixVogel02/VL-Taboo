
import os

import torch
from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset


class LabeledDataset(Dataset):
    """Dataset for image-wise attribute loading of CUB."""

    def __init__(self, img_name_id, id_attribute, img_attr, img_cls, class_names, min_certainty=4,
                 root_dir="/data/felix/new_bachelor/cub/CUB_200_2011/images/", transform=lambda x: x):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        certainties = ["1", "2", "3", "4"]
        self.certainties = certainties[int(min_certainty) - 1:]  # All certainties of attributes that are used.
        self.img_attr = img_attr
        self.img_name_id = img_name_id
        self.id_attribute = id_attribute
        self.img_cls = img_cls
        self.class_names = class_names
        self.root_dir = root_dir
        self.transform = transform
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

        image = Image.open(img_loc).convert("RGB")
        image = self.transform(image)

        img_id = self.img_name_id[img_name]
        img_infos = self.img_attr[img_id]  # list of (attr_id, is_present, certainty_id, time)
        all_attributes = dict()  # Key: attribute type, Value: list of attributes of that type.
        for info in img_infos:
            attr_id, is_present, certainty_id, time = info[0], info[1], info[2], info[3]
            if is_present == "1" and certainty_id in self.certainties:  # (attr_description, attr)
                attr_descr, attr_name = self.id_attribute[attr_id]
                if type(attr_name) == str:
                    attr_name = [attr_name]
                if attr_descr in all_attributes.keys():
                    # attr_name is a list (should normally have only one attribute as element.)
                    all_attributes[attr_descr].extend(attr_name)
                else:
                    all_attributes[attr_descr] = attr_name
        label = self.class_names[self.img_cls[img_id]]
        class_id = self.img_cls[img_id]

        sample = {'image': image, 'attributes': all_attributes,
                  "label_name": label, "class_id": class_id, "image_name": img_name, "img_id": img_id}

        return sample
