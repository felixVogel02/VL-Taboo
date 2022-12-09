import os
import pickle
import time

import clip
import open_clip
import torch
import torchvision.transforms as transforms
from transformers import (FlavaModel, FlavaProcessor)
from labeledDataset import LabeledDataset
from labeledDataset_presaved import LabeledDatasetNew

# from awa2_attributes_vl_checklist import Attributes


class LoadDataset():
    """Predict the attribtues for an image."""

    def __init__(self, model_name="open_clip", device="cuda", datapath='/data/felix/AWA2/Animals_with_Attributes2/', model='ViT-B-32-quickgelu', pretrained='laion400m_e32', new=False):
        """ Takes 11.45 minutes to run."""

        self.base_path = datapath
        if not new:
            start = time.time()
            self.device = device
            # self.model, _, preprocess = open_clip.create_model_and_transforms(
            #     model, pretrained=pretrained, device=self.device)
            valdir = os.path.join(datapath + "JPEGImages/")

            # Load the data:
            # General mapping from idx of class to it's label. Needed because attributes are listed in this order.
            self.idx_label, self.label_idx = self.create_classmapping()
            # Attributes are used by index, therefore this is the mapping we need.
            self.idx_attribute = self.load_attributes()
            self.class_idx_attribute, self.class_idx_attribute_idx = self.map_class_to_attributes(self.idx_attribute)
            self.image_attr = self.load_image_attributes(model_name=model_name)
            self.data_loader, self.dataset, self.model, self.processor = self.load_model(
                self.label_idx, self.class_idx_attribute, self.class_idx_attribute_idx, idx_attribute=self.idx_attribute, image_attr=self.image_attr, model_name=model_name, device="cuda", folder_path="/data/felix/new_bachelor/cub/CUB_200_2011/images")

        else:  # To load already encoded images.
            start = time.time()
            self.device = device
            # self.model, _, preprocess = open_clip.create_model_and_transforms(
            #     model, pretrained=pretrained, device=self.device)
            valdir = os.path.join(datapath + "JPEGImages/")

            # Load the data:
            # General mapping from idx of class to it's label. Needed because attributes are listed in this order.
            self.idx_label, self.label_idx = self.create_classmapping()
            # Attributes are used by index, therefore this is the mapping we need.
            self.idx_attribute = self.load_attributes()
            self.class_idx_attribute, self.class_idx_attribute_idx = self.map_class_to_attributes(self.idx_attribute)
            self.image_attr = self.load_image_attributes(model_name=model_name)
            self.all_imgs_encoded = self.get_img_encoding(model_name=model_name)
            self.data_loader, self.dataset, self.model, self.processor = self.load_model_new(
                self.label_idx, self.all_imgs_encoded, idx_attribute=self.idx_attribute, image_attr=self.image_attr, model_name=model_name, device="cuda")
            # Analysis depends on the experiment.

    def get_img_encoding(self, model_name="open_clip"):
        """Get the image encodings of CUB for the given model."""

        file_path = "/data/felix/new_bachelor/awa2/image_embeddings/" + model_name + "/img_embeddings.pickle"
        with open(file_path, "rb") as act:
            result = pickle.load(act)
        act.close()
        return result

    def tokenize_text(self, raw_text, return_dict=False, tuple_embedding=False):
        """Generate a sentence out of each attribute."""

        # If for each attribtue a positive and negative embedding is given.
        if tuple_embedding:
            with torch.no_grad():
                zeroshot_weights = []
                cnt = 0
                pos_attr_mapping = dict()
                for pos in sorted(raw_text.keys()):
                    texts_pos = open_clip.tokenize(raw_text[pos][0]).to(self.device)  # tokenize
                    texts_neg = open_clip.tokenize(raw_text[pos][1]).to(self.device)  # tokenize
                    class_embeddings_pos = self.model.encode_text(texts_pos)
                    class_embedding_pos = class_embeddings_pos.mean(dim=0)
                    class_embedding_pos /= class_embedding_pos.norm(dim=-1, keepdim=True)

                    class_embeddings_neg = self.model.encode_text(texts_neg)
                    class_embedding_neg = class_embeddings_neg.mean(dim=0)
                    class_embedding_neg /= class_embedding_neg.norm(dim=-1, keepdim=True)
                    zeroshot_weights.append(torch.stack([class_embedding_pos, class_embedding_neg]).to(self.device))
                    pos_attr_mapping[cnt] = pos
                    cnt += 1
                return zeroshot_weights

        for first_val in raw_text.keys():
            break
        if type(raw_text[first_val]) == str:
            text_in_bef = []  # Text input before it is the real input after torch.cat().
            cnt = 0
            pos_attr_mapping = dict()
            for pos in sorted(raw_text.keys()):
                text_in_bef.append(open_clip.tokenize(raw_text[pos]))
                pos_attr_mapping[cnt] = pos
                cnt += 1

            text_input = torch.cat(text_in_bef).to(self.device)
            # Calculate features
            with torch.no_grad():
                text_features = self.model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            if return_dict:
                return text_features, pos_attr_mapping
            return text_features

        elif type(raw_text[first_val]) == list:
            with torch.no_grad():
                zeroshot_weights = []
                cnt = 0
                pos_attr_mapping = dict()
                for pos in sorted(raw_text.keys()):
                    texts = open_clip.tokenize(raw_text[pos]).to(self.device)  # tokenize
                    class_embeddings = self.model.encode_text(texts)
                    class_embedding = class_embeddings.mean(dim=0)
                    #class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
                    class_embedding /= class_embedding.norm(dim=-1, keepdim=True)
                    zeroshot_weights.append(class_embedding)
                    pos_attr_mapping[cnt] = pos
                    cnt += 1
                text_features = torch.stack(zeroshot_weights, dim=1).to(self.device)
                text_features = text_features.T
                if return_dict:
                    return text_features, pos_attr_mapping
                return text_features
        else:
            print("Incorrect tpye of text")
            return None

    def create_classmapping(self):
        """Map the index of a class to it's class name and the other way round."""

        idx_label = dict()
        label_idx = dict()
        with open(self.base_path + "classes.txt", "r") as act:
            line = True
            cnt = 0
            while line:
                line = act.readline()
                line_numb = int(line[:6].strip())  # Starts with 1 not with 0.
                line_label = line[7:].strip()
                # print("Numb: ", line_numb, "Label: ", line_label)
                idx_label[line_numb] = line_label
                label_idx[line_label] = line_numb
                cnt += 1
                if cnt == 50:  # 50 different animal classes.
                    break
        return idx_label, label_idx

    def load_attributes(self):
        """Load the attributes of the dataset."""

        idx_attribute = dict()
        datapath = self.base_path + "/predicates.txt"
        with open(datapath, "r") as act:
            line = True
            cnt = 0
            while line:
                cnt += 1
                line = act.readline()
                line_numb = int(line[:6].strip())
                line_label = line[7:].strip()
                # print("Numb: ", line_numb, "Label: ", line_label)
                idx_attribute[line_numb] = line_label
                if cnt == 85:  # 85 attributes.
                    break
        act.close()
        return idx_attribute

    def map_class_to_attributes(self, idx_attribute):
        """Map the loaded attributes to the different classes."""

        # Load the attributes per class:
        class_idx_attribute = dict()  # Key: class index. Value: List of attribute words.
        class_idx_attribute_idx = dict()  # Key: class index. Value: List of attribute indices.
        with open(self.base_path + "predicate-matrix-binary.txt", "r") as act:
            line = True
            cnt = 0
            while line:
                cnt += 1
                line = act.readline()
                line = line.split()
                # Indices of all attributes for this class. +1 because idx_attribute starts with 0.
                idx = [i + 1 for i, x in enumerate(line) if x == "1"]
                attr = [idx_attribute[i] for i in idx]
                class_idx_attribute[cnt] = attr  # idx_class starts with 1 not with 0.
                class_idx_attribute_idx[cnt] = idx
                if cnt == 50:  # 50 different classes.
                    break
        act.close()
        return class_idx_attribute, class_idx_attribute_idx

    def create_complex_text_labels(self, idx_attribute, text_transformer):
        """Create sentences that should be used for the image-text similarity.
        text_transformer is a function that takes an attribtue and transforms it into a list of sentences."""

        text = dict()  # Dictionairy with index of attribtue as key, and sentence for the attribute as label.
        for key in idx_attribute.keys():
            label = idx_attribute[key]
            sentence = text_transformer(label)
            text[key] = sentence
        return text

    def create_text_labels_each(self, idx_attribute, text_transformer):
        """Creates a list of tuples. For each attribute a tuple with first the positive embeddings (list of embeddings), and second the negative embeddings is created."""

        # Dictionairy with index of attribtue as key, and tuple of sentences (positive, negative) for the attribute as label.
        text = dict()
        for key in idx_attribute.keys():
            label = idx_attribute[key]
            sent1, sent2 = text_transformer(label)
            text[key] = [sent1, sent2]
        return text

    def load_model(self, label_idx, class_idx_attribute, class_idx_attribute_idx, idx_attribute, image_attr, model_name="open_clip", device="cuda"):
        """Load the model and the dataloader."""

        processor = None
        if model_name == "open_clip":
            model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32-quickgelu', pretrained='laion400m_e32', device=device)
        elif model_name == "clip":
            model, preprocess = clip.load('ViT-B/32', device)
        elif model_name == "flava":
            model = FlavaModel.from_pretrained("facebook/flava-full").to(self.device)
            processor = FlavaProcessor.from_pretrained("facebook/flava-full")
            preprocess = transforms.Compose([transforms.PILToTensor()])

        dataset = LabeledDataset(image_attr, label_idx, idx_attribute, transform=preprocess)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1, shuffle=False)  # ,num_workers=0, pin_memory=True)
        return data_loader, dataset, model, processor

    def load_model_new(self, label_idx, all_imgs_encoded, idx_attribute, image_attr, model_name="open_clip", device="cuda", folder_path="/data/felix/new_bachelor/cub/CUB_200_2011/images"):
        """Load the model and the dataloader."""

        processor = None
        if model_name == "open_clip":
            model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32-quickgelu', pretrained='laion400m_e32', device=device)
        elif model_name == "clip":
            model, preprocess = clip.load('ViT-B/32', device)
        elif model_name == "flava":
            model = FlavaModel.from_pretrained("facebook/flava-full").to(self.device)
            processor = FlavaProcessor.from_pretrained("facebook/flava-full")
            preprocess = transforms.Compose([transforms.PILToTensor()])

        valdir = os.path.join(folder_path)
        dataset = LabeledDatasetNew(image_attr, label_idx, idx_attribute, all_imgs_encoded)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1, shuffle=False)  # ,num_workers=0, pin_memory=True)
        return data_loader, dataset, model, processor

    def load_image_attributes(self, model_name="open_clip"):
        """Load the mapping from the image index to the image attributes."""

        if model_name == "open_clip":
            file_path: str = "/home/felix/new_bachelor/awa2/image_annotations/image_attr_raw_similarities_indices.pickle"
            with open(file_path, "rb") as act:
                result = pickle.load(act)
            act.close()
        elif model_name == "clip":
            file_path: str = "/home/felix/new_bachelor/awa2/image_annotations/clip/image_attr_raw_similarities_indices.pickle"
            with open(file_path, "rb") as act:
                result = pickle.load(act)
            act.close()
        elif model_name == "flava":
            file_path: str = "/home/felix/new_bachelor/awa2/image_annotations/flava/image_attr_raw_similarities_indices.pickle"
            with open(file_path, "rb") as act:
                result = pickle.load(act)
            act.close()
        return result


def main():
    torch.cuda.empty_cache()
    # file_path = "/home/felix/new_bachelor/awa2/image_annotations/image_attr_res.pickle"
    # with open(file_path, "rb") as act:
    #     result = pickle.load(act)
    # act.close()
    # new_res = {key: result[key]["raw_similarities"] for key in result.keys()}

    file_path = "/home/felix/new_bachelor/awa2/image_annotations/image_attr_res.pickle"
    with open(file_path, "rb") as act:
        result = pickle.load(act)
    act.close()

    new_res = dict()
    for key in result.keys():
        sim = result[key]["raw_similarities"]
        sim[sim < 0.2] = 0
        indices = torch.nonzero(sim)
        indices = torch.flatten(indices)
        indices = indices + 1  # Because indices of the list start with 0, but attribtue indices with 1.
        new_res[key] = indices

    file_path = "/home/felix/new_bachelor/awa2/image_annotations/image_attr_raw_similarities_indices.pickle"
    with open(file_path, "wb") as act:
        pickle.dump(new_res, act, protocol=pickle.HIGHEST_PROTOCOL)
    act.close()
    # print(new_res)
    # print("Length: ", len(new_res.keys()))


def main1():
    file_path = "/home/felix/new_bachelor/awa2/image_annotations/image_attr_raw_similarities_indices.pickle"
    with open(file_path, "rb") as act:
        new_res = pickle.load(act)
    act.close()
    print(new_res[1])


if __name__ == "__main__":
    main1()
