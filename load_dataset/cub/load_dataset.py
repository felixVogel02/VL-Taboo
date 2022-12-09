import sys

sys.path.append("/home/felix/new_bachelor/cub/")  # noqa

import os
import pickle
import time

import clip
import numpy as np
import open_clip
import torch
import torchvision.transforms as transforms
from natsort import natsorted
from transformers import (FlavaModel, FlavaProcessor)

from labeledDataset import LabeledDataset
from labeledDataset_presaved import LabeledDatasetNew


class LoadDataset():
    """Load the dataset CUB and all the important informations."""

    def __init__(self):
        """Initialize the class."""

        pass

    def do_all(self, avg_factor=1, model_name="open_clip"):
        """Read all important information and create the dataloader and model."""

        img_cls = self.load_image_class()
        img_id_name, img_name_id = self.load_image_id()
        id_attr = self.load_attributes()
        img_attr = self.load_image_attribute_labels()
        certainty = self.load_certainty()
        class_names, names_class = self.load_class_names()
        # class_attributes = self.load_class_labels_over_average(avg_factor=avg_factor)
        descr_indices = self.classify_attr(id_attr)
        # self.keep_certainty3_attributes(img_attr, id_attr)  # Once to create the mapping

        class_attributes = self.load_class_labels_one_per_descr(descr_indices, avg_factor=avg_factor)
        data_loader, dataset, model, processor = self.load_model(img_name_id=img_name_id, id_attribute=id_attr, img_attr=img_attr,
                                                                 img_cls=img_cls, class_names=class_names, model_name=model_name)
        res = {"img_cls": img_cls, "img_id_name": img_id_name, "img_name_id": img_name_id, "id_attr": id_attr,
               "certainty": certainty, "class_names": class_names, "data_loader": data_loader, "dataset": dataset,
               "img_attr": img_attr, "class_attributes": class_attributes, "names_class": names_class, "model": model,
               "processor": processor}
        return res

    def do_all_new(self, avg_factor=1, model_name="open_clip", batch_size=1):
        """Read all important information and create the dataloader and model."""

        img_cls = self.load_image_class()
        # print("Img cls: ", img_cls.keys())
        img_id_name, img_name_id = self.load_image_id()  # 11788
        # img_name_id_check = {key: int(img_name_id[key]) for key in img_name_id.keys()}
        # print("Max, min: ", max(img_name_id_check.values()), min(img_name_id_check.values()))
        id_attr = self.load_attributes()
        img_attr = self.load_image_attribute_labels()
        certainty = self.load_certainty()
        class_names, names_class = self.load_class_names()
        # class_attributes = self.load_class_labels_over_average(avg_factor=avg_factor)
        descr_indices = self.classify_attr(id_attr)
        # self.keep_certainty3_attributes(img_attr, id_attr)  # Once to create the mapping
        img_attr_filtered = self.load_certainty3_attributes()
        all_imgs = self.get_img_encoding(model_name=model_name)

        class_attributes = self.load_class_labels_one_per_descr(descr_indices, avg_factor=avg_factor)
        data_loader, dataset, model, processor = self.load_model_new(
            all_imgs, img_attr_filtered, img_cls, class_names, model_name=model_name, device="cuda", batch_size=batch_size, img_id_name=img_id_name)
        res = {"img_cls": img_cls, "img_id_name": img_id_name, "img_name_id": img_name_id, "id_attr": id_attr,
               "certainty": certainty, "class_names": class_names, "data_loader": data_loader, "dataset": dataset,
               "img_attr": img_attr, "class_attributes": class_attributes, "names_class": names_class, "model": model,
               "processor": processor}
        return res

    def give_dataset_infos(self, avg_factor=1):
        """Read all important information and create the dataloader and model."""

        img_cls = self.load_image_class()
        # print("Img cls: ", img_cls.keys())
        img_id_name, img_name_id = self.load_image_id()  # 11788
        # img_name_id_check = {key: int(img_name_id[key]) for key in img_name_id.keys()}
        # print("Max, min: ", max(img_name_id_check.values()), min(img_name_id_check.values()))
        id_attr = self.load_attributes()
        certainty = self.load_certainty()
        class_names, names_class = self.load_class_names()
        # class_attributes = self.load_class_labels_over_average(avg_factor=avg_factor)
        descr_indices = self.classify_attr(id_attr)
        # self.keep_certainty3_attributes(img_attr, id_attr)  # Once to create the mapping
        img_attr_filtered = self.load_certainty3_attributes()

        class_attributes = self.load_class_labels_one_per_descr(descr_indices, avg_factor=avg_factor)
        res = {"img_cls": img_cls, "img_id_name": img_id_name, "img_name_id": img_name_id, "id_attr": id_attr,
               "certainty": certainty, "class_names": class_names, "class_attributes": class_attributes,
               "names_class": names_class}
        return res

    def get_img_encoding(self, model_name="open_clip"):
        """Get the image encodings of CUB for the given model."""

        file_path = "/data/felix/new_bachelor/cub/image_embeddings/" + model_name + "/img_embeddings.pickle"
        with open(file_path, "rb") as act:
            result = pickle.load(act)
        act.close()
        return result

    def load_certainty3_attributes(self):
        """Load the attribtues saved by self.keep_certainty3_attributes"""

        with open("/data/felix/new_bachelor/cub/felix_image_attribute", "rb") as act:
            result = pickle.load(act)
        act.close()
        return result

    def keep_certainty3_attributes(self, img_attr, id_attribute):
        """Only keep attributes that are present and have a certainty of at least 3.
        For each image id store a dict. The dict has key: Attribute description,
        value: characterisitc of that attribute (should only be one!)"""

        result = dict()
        for img_id in img_attr.keys():
            img_infos = img_attr[img_id]  # list of (attr_id, is_present, certainty_id, time)
            all_attributes = dict()  # Key: attribute type, Value: list of attributes of that type.
            for info in img_infos:
                attr_id, is_present, certainty_id, time = info[0], info[1], info[2], info[3]
                if is_present == "1" and certainty_id in ["3", "4"]:  # (attr_description, attr)
                    attr_descr, attr_name = id_attribute[attr_id]
                    if type(attr_name) == str:
                        attr_name = [attr_name]
                    if attr_descr in all_attributes.keys():
                        # attr_name is a list (should normally have only one attribute as element.)
                        all_attributes[attr_descr].extend(attr_name)
                    else:
                        all_attributes[attr_descr] = attr_name
                    if len(attr_name) > 1:
                        print("Error! Too many attribute characteristics assigned for one attribute description.")
                        print(attr_name)
            result[img_id] = all_attributes
        with open("/data/felix/new_bachelor/cub/felix_image_attribute", "wb") as act:
            pickle.dump(result, act, protocol=pickle.HIGHEST_PROTOCOL)
        act.close()

    def load_attributes(self, file_name: str = "/data/felix/new_bachelor/cub/attributes.txt") -> dict:
        """Load the attributes from the file. Returns a dict with index of the attribute as key,
        and (description, attribute) as value."""

        # For each attribute index a tuple (x, y) with x = attr description, y = manifestation of attribute.
        id_attr = dict()
        with open(file_name, "r") as file_map:
            line = True
            while line:
                line = file_map.readline()
                if len(line) != 0:  # Not an empty line. (Last line is empty and should be avoided.)
                    res = line.split(" ")
                    id = res[0]
                    res = res[1].split("::")
                    attr_description = res[0].replace("_", " ")
                    attr = res[1].replace("_", " ")[:-1]
                    id_attr[id] = (attr_description, attr)
        return id_attr

    def load_image_attribute_labels(self, file_name: str = "/data/felix/new_bachelor/cub/CUB_200_2011/attributes/image_attribute_labels.txt"):
        """Load the mapping from the image id to the attribute id, presence of attribute
        and the certainty of the attributes."""

        img_attr = dict()
        with open(file_name, "r") as file_map:
            line = True
            while line:
                line = file_map.readline()
                if len(line) != 0:  # Not an empty line. (Last line is empty and should be avoided.)
                    res = line.split(" ")
                    img_id = res[0]
                    attr_id = res[1]
                    is_present = res[2]
                    certainty_id = res[3]
                    time = res[4][:-1]  # Remove \n
                    new_tuple = (attr_id, is_present, certainty_id, time)
                    if img_id in img_attr.keys():
                        img_attr[img_id].append(new_tuple)
                    else:
                        img_attr[img_id] = [new_tuple]

        return img_attr

    def load_certainty(self, file_name: str = "/data/felix/new_bachelor/cub/CUB_200_2011/attributes/certainties.txt"):
        """Load the mapping from the certainty index to the certainty name."""

        certainty = dict()
        with open(file_name, "r") as file_map:
            line = True
            while line:
                line = file_map.readline()
                if len(line) != 0:  # Not an empty line. (Last line is empty and should be avoided.)
                    id = line[0]
                    label = line[1:-1]
                    certainty[id] = label

        return certainty

    def load_class_names(self, file_name: str = "/data/felix/new_bachelor/cub/CUB_200_2011/classes.txt"):
        """Loads the mapping with class index as key and class name as value."""

        class_names = dict()
        names_class = dict()
        with open(file_name, "r") as file_map:
            line = True
            while line:
                line = file_map.readline()
                if len(line) != 0:  # Not an empty line. (Last line is empty and should be avoided.)
                    res = line.split(" ")
                    id = res[0]
                    label = res[1][:-1].split(".")[1].replace("_", " ")
                    class_names[id] = label
                    names_class[label] = id

        return class_names, names_class

    def load_image_class(self, file_name: str = "/data/felix/new_bachelor/cub/CUB_200_2011/image_class_labels.txt"):
        """Load for each image index the index of its assigned class."""

        img_cls = dict()
        with open(file_name, "r") as file_map:
            line = True
            while line:
                line = file_map.readline()
                if len(line) != 0:  # Not an empty line. (Last line is empty and should be avoided.)
                    res = line.split(" ")
                    img_id = res[0]
                    cls_id = res[1][:-1]
                    img_cls[img_id] = cls_id

        return img_cls

    def load_image_id(self, file_name: str = "/data/felix/new_bachelor/cub/CUB_200_2011/images.txt"):
        """Load for each image id the file name and vice versa."""

        id_name = dict()
        name_id = dict()
        with open(file_name, "r") as file_map:
            line = True
            while line:
                line = file_map.readline()
                if len(line) != 0:  # Not an empty line. (Last line is empty and should be avoided.)
                    res = line.split(" ")
                    id = res[0]
                    name = res[1][:-1]  # Remove \n
                    id_name[id] = name
                    name_id[name] = id

        return id_name, name_id

    def load_model(self, img_name_id, id_attribute, img_attr, img_cls, class_names, min_certainty="3", model_name="open_clip", device="cuda", folder_path="/data/felix/new_bachelor/cub/CUB_200_2011/images"):
        """Load the model and the dataloader."""

        valdir = os.path.join(folder_path)
        processor = None
        if model_name == "open_clip":
            model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32-quickgelu', pretrained='laion400m_e32', device=device)
        elif model_name == "clip":
            model, preprocess = clip.load('ViT-B/32', device)
        elif model_name == "flava":
            model = FlavaModel.from_pretrained("facebook/flava-full")
            processor = FlavaProcessor.from_pretrained("facebook/flava-full")
            preprocess = transforms.Compose([transforms.PILToTensor()])
        else:  # Other model, f.e. from VLChecklist
            model = None
            preprocess = transforms.Compose([transforms.PILToTensor()])

        dataset = LabeledDataset(img_name_id=img_name_id, id_attribute=id_attribute, img_attr=img_attr,
                                 img_cls=img_cls, class_names=class_names, min_certainty=min_certainty, transform=preprocess)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1, shuffle=False,
            num_workers=1, pin_memory=True)
        return data_loader, dataset, model, processor

    def load_model_new(self, all_imgs, img_attr_filtered, img_cls, class_names, model_name="open_clip", device="cuda", batch_size=1, img_id_name=None):
        """Load the model and the dataloader."""

        processor = None
        if model_name == "open_clip":
            model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32-quickgelu', pretrained='laion400m_e32', device=device)
        elif model_name == "clip":
            model, preprocess = clip.load('ViT-B/32', device)
        elif model_name == "flava":
            model = FlavaModel.from_pretrained("facebook/flava-full")
            processor = FlavaProcessor.from_pretrained("facebook/flava-full")
            preprocess = transforms.Compose([transforms.PILToTensor()])

        dataset = LabeledDatasetNew(all_imgs, img_attr_filtered, img_cls, class_names, img_id_name=img_id_name)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=1, pin_memory=True)
        return data_loader, dataset, model, processor

    def load_class_labels(self, min_rel_occ=70, file_name: str = "/data/felix/new_bachelor/cub/CUB_200_2011/attributes/class_attribute_labels_continuous.txt"):
        """Load per-class attributes instead of per-image attributes."""

        # Key: id of the class, value: list of attribute indices of attributes that occur in enough images.
        id_name = dict()
        with open(file_name, "r") as file_map:
            line = True
            cnt = 0
            while line:
                cnt += 1
                # print("Count: ", cnt)
                line = file_map.readline()
                found_attr = []
                if len(line) != 0:  # Not an empty line. (Last line is empty and should be avoided.)
                    res = line.split(" ")
                    for idx in range(len(res)):
                        val = res[idx]
                        # print("Red: ", val)
                        if val == ".":  # Some nubmers start with "." instead of "0."
                            val = "0" + val
                        val = float(val)
                        # print("Value: ", val)
                        if val > min_rel_occ:
                            # print("Value: ", val)
                            found_attr.append(idx + 1)  # Attribute indices start with 1, but list indices with 0.

                    id_name[cnt] = found_attr

        return id_name

    def load_class_labels_over_average(self, avg_factor=1, file_name: str = "/data/felix/new_bachelor/cub/CUB_200_2011/attributes/class_attribute_labels_continuous.txt"):
        """Load per-class attributes instead of per-image attributes. ONly take the attributes of each class, that
        occure more often than the average of the attribtues of that class (0-occurrences excluded."""

        # Key: id of the class, value: list of attribute indices of attributes that occur in enough images.
        id_name = dict()
        with open(file_name, "r") as file_map:
            line = True
            cnt = 0
            while line:
                cnt += 1
                # print("Count: ", cnt)
                line = file_map.readline()
                if len(line) != 0:  # Not an empty line. (Last line is empty and should be avoided.)
                    attr_dict = dict()  # Key: id of attribute, value: percentage of the attribute that it is correct.
                    res = line.split(" ")
                    avg = 0
                    numb = 0
                    for idx in range(len(res)):
                        val = res[idx]
                        # print("Red: ", val)
                        if val == ".":  # Some nubmers start with "." instead of "0."
                            val = "0" + val
                        val = float(val)
                        if val > 0.0001:  # Don't consider 0-percentage values.
                            attr_dict[idx + 1] = val
                            numb += 1
                            avg += val
                    avg = avg / numb
                    # print("Average: ", avg)
                    found_attr = []
                    for key in attr_dict.keys():
                        if attr_dict[key] > avg_factor * avg:
                            found_attr.append(key)

                    id_name[cnt] = found_attr

        return id_name

    def classify_attr(self, id_attr):
        """Creates a dict that maps from the attribute description to all the attribtues according to that description."""

        descr_indices = dict()  # Key: attr_descr, value: list of ids of attributes with the same attribute description.
        found_descr = []
        for id in id_attr.keys():
            attr_descr = id_attr[id][0]
            attr = id_attr[id][1]
            if attr_descr in descr_indices.keys():
                descr_indices[attr_descr].append(id)
            else:
                descr_indices[attr_descr] = [id]
        return descr_indices

    def load_class_labels_one_per_descr(self, descr_indices, avg_factor=1, file_name: str = "/data/felix/new_bachelor/cub/CUB_200_2011/attributes/class_attribute_labels_continuous.txt"):
        """Load per-class attributes instead of per-image attributes. ONly take the attributes of each class, that
        occure more often than the average of the attribtues of that class (0-occurrences excluded."""

        # Key: id of the class, value: list of attribute indices of attributes that occur in enough images.
        id_name = dict()
        # print(descr_indices)
        with open(file_name, "r") as file_map:
            line = True
            cnt = 0
            while line:
                cnt += 1
                # print("Count: ", cnt)
                line = file_map.readline()
                if len(line) != 0:  # Not an empty line. (Last line is empty and should be avoided.)
                    attr_dict = dict()  # Key: id of attribute, value: percentage of the attribute that it is correct.
                    res = line.split(" ")
                    avg = 0
                    numb = 0
                    for idx in range(len(res)):
                        val = res[idx]
                        # print("Red: ", val)
                        if val == ".":  # Some nubmers start with "." instead of "0."
                            val = "0" + val
                        val = float(val)
                        if val > 0.0001:  # Don't consider 0-percentage values.
                            attr_dict[idx + 1] = val
                            numb += 1
                            avg += val
                    avg = avg_factor * avg / numb

                    # Take the best attribute (maximum 1) of each attribute description that is over the average.
                    found_attr = []
                    for attr_descr in descr_indices.keys():
                        fitting_ids = []
                        probs = []
                        for id in attr_dict.keys():
                            if str(id) in descr_indices[attr_descr] and attr_dict[id] > avg:
                                fitting_ids.append(id)
                                probs.append(attr_dict[id])
                        # Take the best fitting id.
                        probs = np.array(probs)
                        if len(fitting_ids) > 0:
                            id_idx = np.argmax(probs)
                            found_attr.append(fitting_ids[id_idx])

                    id_name[cnt] = found_attr

        return id_name


def main():
    loader = LoadDataset()
    img_cls = loader.load_image_class()
    img_id_name, img_name_id = loader.load_image_id()
    id_attr = loader.load_attributes()
    img_attr = loader.load_image_attribute_labels()
    certainty = loader.load_certainty()
    class_names, names_class = loader.load_class_names()
    data_loader, dataset, model = loader.load_model(img_name_id=img_name_id, id_attribute=id_attr, img_attr=img_attr,
                                                    img_cls=img_cls, class_names=class_names, model_name="open_clip")


def test(root_dir="/data/felix/new_bachelor/cub/CUB_200_2011/images/"):
    all_folders = os.listdir(root_dir)
    all_imgs = []
    for folder in all_folders:
        all_imgs.extend([os.path.join(folder, i) for i in os.listdir(os.path.join(root_dir, folder))])
    all_imgs = natsorted(all_imgs)
    print(all_imgs)


if __name__ == "__main__":
    main()
    # test()
