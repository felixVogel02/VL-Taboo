import torch
from torch.utils.data import Dataset


class LabeledDatasetNew(Dataset):
    """Dataset for image-wise attribute loading of CUB."""

    def __init__(self, all_imgs, img_attr_filtered, img_cls, class_names, img_id_name):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.all_imgs = all_imgs
        self.img_attr_filtered = img_attr_filtered
        self.class_names = class_names
        self.img_cls = img_cls

        # Just to get the image name for the given id back.
        self.img_id_name = img_id_name

    def __len__(self):
        return len(self.img_attr_filtered.keys())

    def __getitem__(self, id):
        if torch.is_tensor(id):
            img_id = img_id.tolist()
        # print("ID: ", id)
        img_id = id + 1

        img = self.all_imgs[id]  # Indices of all_imgs list start with 0 but img_id with index 1.
        # Key: attribute type, Value: list of attributes of that type.
        all_attributes = self.img_attr_filtered[str(img_id)]
        label = self.class_names[self.img_cls[str(img_id)]]
        class_id = self.img_cls[str(img_id)]

        img_name = self.img_id_name[str(img_id)]

        sample = {'image': img, 'attributes': all_attributes,
                  "label_name": label, "class_id": class_id, "img_id": img_id, "img_name": img_name}

        return sample
