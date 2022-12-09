import copy
import os
import pickle
import random
import ssl
import time
from cgi import test
from sys import getsizeof

import clip
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import pandas as pd
import seaborn as sn
import torch
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from matplotlib.ticker import MaxNLocator
from regex import B
from sklearn import metrics
from sklearn.metrics import confusion_matrix

from utilities.analysis import Analyzer
from sentenceCreator.awa2_attributes import Attributes

# from awa2_attributes_vl_checklist import Attributes


class AttributePredictor():
    """Predict the attributes for an image."""

    def __init__(self, device="cuda", datapath='/data/felix/AWA2/Animals_with_Attributes2/JPEGImages/', model='ViT-B-32-quickgelu', pretrained='laion400m_e32'):
        """ Takes 11.45 minutes to run."""

        start = time.time()
        self.device = device
        self.model, _, preprocess = open_clip.create_model_and_transforms(
            model, pretrained=pretrained, device=self.device)
        valdir = os.path.join(datapath)
        self.val_dataset = datasets.ImageFolder(valdir, preprocess)
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=1, shuffle=False,
            num_workers=1, pin_memory=True)

        # Load the data:
        # General mapping from idx of class to it's label. Needed because attributes are listed in this order.
        self.idx_label, self.label_idx = self.create_classmapping()
        # Images are loaded with different indices, therefore they are mapped individually to their labels.
        self.image_idx_label, self.image_label_idx = self.create_image_idx_mapping()
        # Attributes are used by index, therefore this is the mapping we need.
        self.idx_attribute = self.load_attributes()
        self.class_idx_attribute, self.class_idx_attribute_idx = self.map_class_to_attributes(self.idx_attribute)

        # Analysis depends on the experiment.

    def create_classmapping(self):
        """Map the index of a class to it's class name and the other way round."""

        idx_label = dict()
        label_idx = dict()
        with open("/data/felix/AWA2/Animals_with_Attributes2/classes.txt", "r") as act:
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

    def create_image_idx_mapping(self):
        """Loaded images have different idices than the indices we loaded with the .txt files. Therefore we need their mapping as well."""

        image_idx_label = dict()
        labeled_classes = self.val_dataset.class_to_idx
        for key in labeled_classes.keys():
            image_idx_label[labeled_classes[key]] = key
        return image_idx_label, self.val_dataset.class_to_idx

    def load_attributes(self, datapath="/data/felix/AWA2/Animals_with_Attributes2/predicates.txt"):
        """Load the attributes of the dataset."""

        idx_attribute = dict()
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
        with open("/data/felix/AWA2/Animals_with_Attributes2/predicate-matrix-binary.txt", "r") as act:
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

    def create_simple_text_labels(self, idx_attribute):
        """Create sentences that should be used for the image-text similarity. Every attribtue uses the same single sentence."""

        text = dict()  # Dictionairy with index of attribtue as key, and sentence for the attribute as label.
        for key in idx_attribute.keys():
            label = idx_attribute[key]
            sentence = f"This is a photo of a {label} animal."
            text[key] = sentence
        return text

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

    def validate_attributes1(self, text_features, folder="", max_attr_numb=85):
        """Predict the attributes for every image."""

        max_attr_numb += 1  # Because in range you need +1.
        result_dict = {class_idx: {attr_idx: 0 for attr_idx in list(
            range(1, max_attr_numb)) + ["count"] + ["percentage_distribution"]} for class_idx in range(1, 51)}
        result_dict["total_result"] = {attr_idx: 0 for attr_idx in list(
            range(1, max_attr_numb)) + ["count"] + ["percentage_distribution"]}
        cnt = 0
        for img, label in self.val_loader:
            cnt += 1
            if cnt < 0:  # Don't break.
                break

            with torch.no_grad():
                image_features = self.model.encode_image(img.to(self.device))

            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            # Label:
            idx_label = self.label_idx[self.image_idx_label[label[0].item()]]  # Class index of the label class.
            idx_label_list = self.class_idx_attribute_idx[idx_label]  # List of attribute indices

            # Top 1 analysis:
            # values, indices = similarity[0].topk(1)
            # # Index of the attribute that is predicted.
            # idx_predict = indices[0].item() + 1  # Class indices start with 1. But order of the sentences with 0.
            idx_label_set = set(idx_label_list)
            for key in range(1, max_attr_numb):
                values, indices = similarity[0].topk(key)
                idx_predict_set = set([indices[i].item() + 1 for i in range(key)])
                intersection = len(idx_predict_set.intersection(idx_label_set))
                result_dict[idx_label][key] += intersection
                result_dict["total_result"][key] += intersection
            result_dict[idx_label]["count"] += 1
            result_dict["total_result"]["count"] += 1
            # Also save distribution of the probability over the predicted attributes.
            values, indices = similarity[0].topk(85)
            result_dict[idx_label]["percentage_distribution"] += values
            result_dict["total_result"]["percentage_distribution"] += values

        # For each class we save for each number of taken attributes a tuple with two elements:
        # First: percentage of taken attributes of the top x, ABsolute nubmer of taken attribtues of the top x.
        relative_results = copy.deepcopy(result_dict)
        for i in range(1, 51):
            class_examples = relative_results[i]["count"]
            relative_results[i]["percentage_distribution"] = relative_results[i]["percentage_distribution"] / class_examples
            for attr_num in range(1, max_attr_numb):
                relative_results[i][attr_num] = (
                    relative_results[i][attr_num] / (class_examples * attr_num), relative_results[i][attr_num] / class_examples)
        numb = relative_results["total_result"]["count"]
        relative_results["total_result"]["percentage_distribution"] = relative_results["total_result"]["percentage_distribution"] / numb
        for attr_num in range(1, max_attr_numb):
            relative_results["total_result"][attr_num] = (
                relative_results["total_result"][attr_num] / (numb * attr_num), relative_results["total_result"][attr_num] / numb)

        with open("/home/felix/new_bachelor/awa2/results/first_experiment/" + folder + "absolute_results.pickle", "wb") as act:
            pickle.dump(result_dict, act, protocol=pickle.HIGHEST_PROTOCOL)
        act.close()
        with open("/home/felix/new_bachelor/awa2/results/first_experiment/" + folder + "relative_results.pickle", "wb") as act:
            pickle.dump(relative_results, act, protocol=pickle.HIGHEST_PROTOCOL)
        act.close()

    def validate_attributes1_per_attr(self, text_features):
        """Predict the attributes for every image."""

        result_dict = {attr_idx: {"correct": 0, "count": 0} for attr_idx in range(1, 86)}
        cnt = 0
        for img, label in self.val_loader:
            cnt += 1
            if cnt < 0:  # Don't break.
                break

            with torch.no_grad():
                image_features = self.model.encode_image(img.to(self.device))

            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            # Label:
            idx_label = self.label_idx[self.image_idx_label[label[0].item()]]  # Class index of the label class.
            idx_label_list = self.class_idx_attribute_idx[idx_label]  # List of attribute indices

            # Top 1 analysis:
            # values, indices = similarity[0].topk(1)
            # # Index of the attribute that is predicted.
            # idx_predict = indices[0].item() + 1  # Class indices start with 1. But order of the sentences with 0.
            idx_label_set = set(idx_label_list)

            values, indices = similarity[0].topk(1)
            idx_predict = indices[0].item() + 1
            idx_predict_set = set([idx_predict])
            intersection = len(idx_predict_set.intersection(idx_label_set))
            result_dict[idx_predict]["correct"] += intersection
            result_dict[idx_predict]["count"] += 1

        # For each class we save for each number of taken attributes a tuple with two elements:
        # First: percentage of taken attributes of the top x, ABsolute nubmer of taken attribtues of the top x.
        relative_results = copy.deepcopy(result_dict)
        for attr_idx in range(1, 86):
            relative_results[attr_idx]["accuracy"] = relative_results[attr_idx]["correct"] / \
                relative_results[attr_idx]["count"]

        with open("/home/felix/new_bachelor/awa2/results/first_experiment/experiment1_per_attr_absolute_results.pickle", "wb") as act:
            pickle.dump(result_dict, act, protocol=pickle.HIGHEST_PROTOCOL)
        act.close()
        with open("/home/felix/new_bachelor/awa2/results/first_experiment/experiment1_per_attr_relative_results.pickle", "wb") as act:
            pickle.dump(relative_results, act, protocol=pickle.HIGHEST_PROTOCOL)
        act.close()

    def validate_attributes2(self, text_features, folder=""):
        """Predict the attributes for every image."""

        # key: (x, y, z, k): key = Percentage (5, ..., 100) the first attributes should achieve in minimum.
        # x = Percentage actually achieved added; y = Number of attributes found; z = Number of correct attributes; k = Percent of correct predicted attributes
        result_dict = {class_idx: {**{attr_idx: (0, 0, 0, 0) for attr_idx in list(
            range(0, 101, 1))},  **{"count": 0}} for class_idx in range(1, 51)}
        result_dict["total_result"] = {**{attr_idx: (0, 0, 0, 0)
                                          for attr_idx in list(range(0, 101, 1))}, **{"count": 0}}
        cnt = 0
        for img, label in self.val_loader:
            cnt += 1
            if cnt < 0:  # Don't break.
                break

            with torch.no_grad():
                image_features = self.model.encode_image(img.to(self.device))

            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            # Label:
            idx_label = self.label_idx[self.image_idx_label[label[0].item()]]  # Class index of the label class.
            idx_label_list = self.class_idx_attribute_idx[idx_label]  # List of attribute indices

            # Save data:
            idx_label_set = set(idx_label_list)
            for key in range(0, 101, 1):
                values, indices = similarity[0].topk(85)
                key_perc = key / 100  # Percentage that should be achieved in sum as minimum.

                inner_cnt = 0
                curr_sum = 0
                for i in values:
                    inner_cnt += 1
                    curr_sum += i
                    if curr_sum >= key_perc:
                        break

                idx_predict_set = set([indices[i].item() + 1 for i in range(inner_cnt)])
                intersection = len(idx_predict_set.intersection(idx_label_set))
                a = result_dict[idx_label][key][0] + curr_sum
                b = result_dict[idx_label][key][1] + len(idx_predict_set)
                c = result_dict[idx_label][key][2] + intersection
                d = result_dict[idx_label][key][3] + intersection / len(idx_predict_set)
                result_dict[idx_label][key] = (a, b, c, d)
                aa = result_dict["total_result"][key][0] + curr_sum
                bb = result_dict["total_result"][key][1] + len(idx_predict_set)
                cc = result_dict["total_result"][key][2] + intersection
                dd = result_dict["total_result"][key][3] + intersection / len(idx_predict_set)
                result_dict["total_result"][key] = (aa, bb, cc, dd)
            result_dict[idx_label]["count"] += 1
            result_dict["total_result"]["count"] += 1

        # For each class we save for each number of taken attributes a tuple with two elements:
        # First: percentage of taken attributes of the top x, ABsolute nubmer of taken attribtues of the top x.
        relative_results = copy.deepcopy(result_dict)
        for i in range(1, 51):
            numb = relative_results[i]["count"]
            for perc in range(0, 101, 1):
                relative_results[i][perc] = (relative_results[i][perc][0] / numb, relative_results[i][perc]
                                             [1] / numb, relative_results[i][perc][2] / numb, relative_results[i][perc][3] / numb)
        numb = relative_results["total_result"]["count"]
        for perc in range(0, 101, 1):
            relative_results["total_result"][perc] = (relative_results["total_result"][perc][0] / numb, relative_results["total_result"]
                                                      [perc][1] / numb, relative_results["total_result"][perc][2] / numb, relative_results["total_result"][perc][3] / numb)

        with open("/home/felix/new_bachelor/awa2/results/first_experiment/top_percentage/" + folder + "absolute_results.pickle", "wb") as act:
            pickle.dump(result_dict, act, protocol=pickle.HIGHEST_PROTOCOL)
        act.close()
        with open("/home/felix/new_bachelor/awa2/results/first_experiment/top_percentage/" + folder + "relative_results.pickle", "wb") as act:
            pickle.dump(relative_results, act, protocol=pickle.HIGHEST_PROTOCOL)
        act.close()

    def validate_category3(self, text_features, text_feature_mapping, folder="", max_attr_numb=False):
        """Predict the attributes for every image out of a category of attributes."""

        if not max_attr_numb:
            max_attr_numb = len(text_feature_mapping.keys())
        max_attr_numb += 1  # Because in range you need +1.

        result_dict = {text_feature_mapping[key]: {"top-1-accuracy": 0, "count": 0}
                       for key in text_feature_mapping.keys()}
        result_dict["total_result"] = {attr_idx: 0 for attr_idx in list(
            range(1, max_attr_numb)) + ["count"] + ["percentage_distribution"]}
        cnt = 0
        for img, label in self.val_loader:
            cnt += 1
            if cnt < 0:  # Don't break.
                break

            with torch.no_grad():
                image_features = self.model.encode_image(img.to(self.device))

            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            # Label:
            idx_label = self.label_idx[self.image_idx_label[label[0].item()]]  # Class index of the label class.
            idx_label_list = self.class_idx_attribute_idx[idx_label]  # List of attribute indices

            # Top 1 analysis:
            # values, indices = similarity[0].topk(1)
            # # Index of the attribute that is predicted.
            # idx_predict = indices[0].item() + 1  # Class indices start with 1. But order of the sentences with 0.
            idx_label_set = set(idx_label_list)
            for key in range(1, max_attr_numb):
                values, indices = similarity[0].topk(key)
                idx_predict_set = set([text_feature_mapping[indices[i].item()] for i in range(key)])
                intersection = len(idx_predict_set.intersection(idx_label_set))
                result_dict["total_result"][key] += intersection
            result_dict["total_result"]["count"] += 1
            # Also save distribution of the probability over the predicted attributes.
            values, indices = similarity[0].topk(len(text_feature_mapping.keys()))
            result_dict["total_result"]["percentage_distribution"] += values

            idx_predict_set = set([text_feature_mapping[indices[0].item()]])
            intersection = len(idx_predict_set.intersection(idx_label_set))

            # Is this really accuracy?!
            result_dict[text_feature_mapping[indices[0].item()]]["top-1-accuracy"] += intersection  # 0 or 1.
            result_dict[text_feature_mapping[indices[0].item()]]["count"] += 1

        # For each class we save for each number of taken attributes a tuple with two elements:
        # First: percentage of taken attributes of the top x, ABsolute nubmer of taken attribtues of the top x.
        relative_results = copy.deepcopy(result_dict)
        for _key in text_feature_mapping.keys():
            key = text_feature_mapping[_key]
            relative_results[key]["top-1-accuracy"] = relative_results[key]["top-1-accuracy"] / \
                relative_results[key]["count"]

        numb = relative_results["total_result"]["count"]
        relative_results["total_result"]["percentage_distribution"] = relative_results["total_result"]["percentage_distribution"] / numb
        for attr_num in range(1, max_attr_numb):
            relative_results["total_result"][attr_num] = (
                relative_results["total_result"][attr_num] / (numb * attr_num), relative_results["total_result"][attr_num] / numb)

        with open("/home/felix/new_bachelor/awa2/results/predict_category/" + folder + "absolute_results.pickle", "wb") as act:
            pickle.dump(result_dict, act, protocol=pickle.HIGHEST_PROTOCOL)
        act.close()
        with open("/home/felix/new_bachelor/awa2/results/predict_category/" + folder + "relative_results.pickle", "wb") as act:
            pickle.dump(relative_results, act, protocol=pickle.HIGHEST_PROTOCOL)
        act.close()

    def validate4(self, text_features_list, folder=""):
        """Analyse each attribute on it's own by comparing each attribtue with it's negated form."""

        result_dict = {i: {"true_positive": 0, "false_positive": 0, "true_negative": 0, "false_negative": 0, "count": 0}
                       for i in range(1, 86)}
        cnt = 0
        for img, label in self.val_loader:
            cnt += 1
            if cnt < 0:  # Don't break.
                break

            with torch.no_grad():
                image_features = self.model.encode_image(img.to(self.device))

            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Label:
            idx_label = self.label_idx[self.image_idx_label[label[0].item()]]  # Class index of the label class.
            idx_label_list = self.class_idx_attribute_idx[idx_label]  # List of attribute indices
            idx_label_set = set(idx_label_list)

            # We have 85 pairs of attributes and their negated form. Go through this pairs.
            # The first element is the attribute, the second the negated attribute for each pair.
            for _idx, text_features in enumerate(text_features_list):
                idx = _idx + 1  # Attribute list starts with index 1.
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

                # Top 1 analysis:
                values, indices = similarity[0].topk(1)
                # Index of the attribute that is predicted.
                # If 0, attribute with index idx is predicted, else (1) it's negated form is predicted.
                attr_predict = indices[0].item()
                # True if attribute is correct (in label attribute set) else false.
                attr_label = bool(idx in idx_label_list)
                if attr_predict == 0 and attr_label:
                    result_dict[idx]["true_positive"] += 1
                elif attr_predict == 0 and not attr_label:
                    result_dict[idx]["false_positive"] += 1
                elif attr_predict == 1 and attr_label:
                    result_dict[idx]["false_negative"] += 1
                elif attr_predict == 1 and not attr_label:
                    result_dict[idx]["true_negative"] += 1
                result_dict[idx]["count"] += 1

        # For each class we save for each number of taken attributes a tuple with two elements:
        # First: percentage of taken attributes of the top x, ABsolute nubmer of taken attribtues of the top x.
        relative_results = copy.deepcopy(result_dict)
        for key in relative_results.keys():
            relative_results[key]["accuracy"] = (relative_results[key]["true_positive"] + relative_results[key]["true_negative"]) / \
                relative_results[key]["count"]

        with open("/home/felix/new_bachelor/awa2/results/predict_attribute/" + folder + "absolute_results.pickle", "wb") as act:
            pickle.dump(result_dict, act, protocol=pickle.HIGHEST_PROTOCOL)
        act.close()
        with open("/home/felix/new_bachelor/awa2/results/predict_attribute/" + folder + "relative_results.pickle", "wb") as act:
            pickle.dump(relative_results, act, protocol=pickle.HIGHEST_PROTOCOL)
        act.close()

    def count_class_samples(self):
        """For each class count the number of samples in the dataset awa2."""

        class_cnt = {"class": {i: 0 for i in range(1, 51)},
                     "samples": 0}
        for img, label in self.val_loader:
            idx_label = self.label_idx[self.image_idx_label[label[0].item()]]  # Class index of the label class.
            class_cnt["class"][idx_label] += 1
            class_cnt["samples"] += 1
        print(class_cnt)
        with open("/home/felix/new_bachelor/awa2/dataset_analysis/class_occurrences.pickle", "wb") as act:
            pickle.dump(class_cnt, act, protocol=pickle.HIGHEST_PROTOCOL)
        act.close()
        return class_cnt


def experiment1(simple_embedding=True, folder="", max_attr_numb=85):
    """Implementation of the first experiment, where the top-x attribute predictions are checked if they are in the correct class."""

    start = time.time()
    attrPred = AttributePredictor()
    analyzer = Analyzer()
    if not simple_embedding:
        attr_converter_class = Attributes()
        # print(getsizeof(attr_converter_class))
        raw_text = attrPred.create_complex_text_labels(attrPred.idx_attribute, attr_converter_class.attribute_to_text)
        #raw_text = attrPred.create_simple_text_labels(attrPred.idx_attribute)
    else:
        raw_text = attrPred.create_simple_text_labels(attrPred.idx_attribute)
    text_features = attrPred.tokenize_text(raw_text)
    print("Text tokenized")
    acc = attrPred.validate_attributes1(text_features, folder=folder, max_attr_numb=max_attr_numb)
    print("Predictions made")
    analyzer.experiment_analysis1(folder=folder, max_attr_numb=max_attr_numb)
    end = time.time()
    print("Time consumption: ", end - start)


def experiment1_per_attr_analysis(simple_embedding=False):
    """Analysis the results of experiment1 per attribute."""

    start = time.time()
    attrPred = AttributePredictor()
    analyzer = Analyzer()
    if not simple_embedding:
        attr_converter_class = Attributes()
        # print(getsizeof(attr_converter_class))
        raw_text = attrPred.create_complex_text_labels(attrPred.idx_attribute, attr_converter_class.attribute_to_text)
        #raw_text = attrPred.create_simple_text_labels(attrPred.idx_attribute)
    else:
        raw_text = attrPred.create_simple_text_labels(attrPred.idx_attribute)
    # text_features = attrPred.tokenize_text(raw_text)
    # print("Text tokenized")
    # acc = attrPred.validate_attributes1_per_attr(text_features)
    print("Predictions made")
    analyzer.experiment1_analysis_per_attr(attrPred.idx_attribute)
    end = time.time()
    print("Time consumption: ", end - start)


def experiment2(simple_embedding=True, folder=""):
    """Implementation of the second experiment, where the top-x percentage attribute predictions are checked if they are in the correct class."""

    start = time.time()
    attrPred = AttributePredictor()
    analyzer = Analyzer()
    if not simple_embedding:
        attr_converter_class = Attributes()
        raw_text = attrPred.create_complex_text_labels(attrPred.idx_attribute, attr_converter_class.attribute_to_text)
    else:
        raw_text = attrPred.create_simple_text_labels(attrPred.idx_attribute)
    print("Text tokenized")
    text_features = attrPred.tokenize_text(raw_text)
    acc = attrPred.validate_attributes2(text_features, folder=folder)
    analyzer.experiment_analysis2(folder=folder)
    end = time.time()
    print("Time consumption: ", end - start)


def experiment_one_category3(category, simple_embedding=True, folder="", max_attr_numb=False):
    """Analyse a category on it's own."""

    start = time.time()
    attrPred = AttributePredictor()
    analyzer = Analyzer()
    attr_converter_class = Attributes()
    category_idx_attribute = attr_converter_class.keep_category(category, copy.deepcopy(attrPred.idx_attribute))

    if not simple_embedding:
        raw_text = attrPred.create_complex_text_labels(category_idx_attribute, attr_converter_class.attribute_to_text)
        #raw_text = attrPred.create_simple_text_labels(attrPred.idx_attribute)
    else:
        raw_text = attrPred.create_simple_text_labels(category_idx_attribute)
    # pos_attr_mapping: Maps from the position of the attribtue in the text_features (0, ...) to the index of the attribute.
    text_features, pos_attr_mapping = attrPred.tokenize_text(raw_text, return_dict=True)
    print("Text tokenized")
    analyzer.experiment_analysis_category3(folder=folder, category=category, used_attr_idxs=list(
        pos_attr_mapping.values()), attr_idx_label_mapping=attrPred.idx_attribute)
    end = time.time()
    print("Time consumption: ", end - start)


def experiment_all_category3(simple_embedding=True, max_attr_numb=False):
    """Analyse a category on it's own."""

    start = time.time()
    attrPred = AttributePredictor()
    analyzer = Analyzer()
    attr_converter_class = Attributes()
    with open("/home/felix/new_bachelor/awa2/dataset_analysis/class_occurrences.pickle", "rb") as act:
        class_cnt = pickle.load(act)
    act.close()
    categories = list(attr_converter_class.attribute_dict.keys())
    for category in categories:
        if not os.path.isdir("/home/felix/new_bachelor/awa2/results/predict_category/new_categories/" + category):
            os.mkdir("/home/felix/new_bachelor/awa2/results/predict_category/new_categories/" + category)
        folder = category + "/"
        category_idx_attribute = attr_converter_class.keep_category(category, copy.deepcopy(attrPred.idx_attribute))

        if not simple_embedding:
            raw_text = attrPred.create_complex_text_labels(
                category_idx_attribute, attr_converter_class.attribute_to_text)
            #raw_text = attrPred.create_simple_text_labels(attrPred.idx_attribute)
        else:
            raw_text = attrPred.create_simple_text_labels(category_idx_attribute)
        # pos_attr_mapping: Maps from the position of the attribtue in the text_features (0, ...) to the index of the attribute.
        text_features, pos_attr_mapping = attrPred.tokenize_text(raw_text, return_dict=True)
        print("Text tokenized")
        attrPred.validate_category3(text_features, pos_attr_mapping, folder="new_categories/" + folder)

        # Find average random accuracy for category:
        category_attrs = list(category_idx_attribute.values())
        result = analyzer.check_if_class_has_attribute(category_attrs, attrPred.class_idx_attribute, class_cnt)
        print("Category: ", category, " Result: ", result)
        avg_prob = result["average_occurrence"] / len(category_attrs)

        # Plot the results.
        analyzer.experiment_analysis_category3(folder="new_categories/" + folder, category=category, used_attr_idxs=list(
            pos_attr_mapping.values()), attr_idx_label_mapping=attrPred.idx_attribute, avg_prob=avg_prob)
        end = time.time()
        print(category, " finished.")
        print("Time consumption: ", end - start)


def analyze_categories_relative(new_categories=False):
    """Analyze the performance per category relative to random guessing."""

    start = time.time()
    attrPred = AttributePredictor()
    analyzer = Analyzer()
    attr_converter_class = Attributes()
    with open("/home/felix/new_bachelor/awa2/dataset_analysis/class_occurrences.pickle", "rb") as act:
        class_cnt = pickle.load(act)
    act.close()
    categories = list(attr_converter_class.attribute_dict.keys())
    if "extra" in categories:
        categories.remove("extra")
    for category in categories:
        folder = category + "/"

        category_idx_attribute = attr_converter_class.keep_category(category, copy.deepcopy(attrPred.idx_attribute))
        category_attrs = list(category_idx_attribute.values())
        result = analyzer.check_if_class_has_attribute(category_attrs, attrPred.class_idx_attribute, class_cnt)
        avg_prob = result["average_occurrence"] / len(category_attrs)
        analyzer.category_peformance_against_random_guessing(folder, avg_prob, new_categories=new_categories)


def analyze_shape_adj(simple_embedding=True, max_attr_numb=False):
    """Analyse a category on it's own."""

    start = time.time()
    attrPred = AttributePredictor()
    analyzer = Analyzer()
    attr_converter_class = Attributes()
    with open("/home/felix/new_bachelor/awa2/dataset_analysis/class_occurrences.pickle", "rb") as act:
        class_cnt = pickle.load(act)
    act.close()
    categories = list(attr_converter_class.attribute_dict.keys())
    category = "shape_adj"
    for left_out_attr in attr_converter_class.shape_adj:
        category = "shape_adj"
        left_out_attr = "only_bipedal_lean_weak"
        if not os.path.isdir("/home/felix/new_bachelor/awa2/results/predict_category/shape_adj_analysis/" + left_out_attr):
            os.mkdir("/home/felix/new_bachelor/awa2/results/predict_category/shape_adj_analysis/" + left_out_attr)
        folder = left_out_attr + "/"
        category_idx_attribute = attr_converter_class.keep_category(
            category, copy.deepcopy(attrPred.idx_attribute), leave_out=[])

        if not simple_embedding:
            raw_text = attrPred.create_complex_text_labels(
                category_idx_attribute, attr_converter_class.attribute_to_text)
            #raw_text = attrPred.create_simple_text_labels(attrPred.idx_attribute)
        else:
            raw_text = attrPred.create_simple_text_labels(category_idx_attribute)
        # pos_attr_mapping: Maps from the position of the attribtue in the text_features (0, ...) to the index of the attribute.
        text_features, pos_attr_mapping = attrPred.tokenize_text(raw_text, return_dict=True)
        print("Text tokenized")
        attrPred.validate_category3(text_features, pos_attr_mapping, folder="shape_adj_analysis/" + folder)

        # Find average random accuracy for category:
        category_attrs = list(category_idx_attribute.values())
        result = analyzer.check_if_class_has_attribute(category_attrs, attrPred.class_idx_attribute, class_cnt)
        print("Category: ", category, " Result: ", result)
        avg_prob = result["average_occurrence"] / len(category_attrs)

        # Plot the results.
        analyzer.experiment_analysis_category3(folder="shape_adj_analysis/" + folder, category=category, used_attr_idxs=list(
            pos_attr_mapping.values()), attr_idx_label_mapping=attrPred.idx_attribute, avg_prob=avg_prob)
        end = time.time()
        print(category, " finished.")
        print("Time consumption: ", end - start)
        break


def check_attr_occurrence_in_class():
    """Check in how many classes attributes of each category really are."""

    start = time.time()
    attrPred = AttributePredictor()
    analyzer = Analyzer()
    attr_converter_class = Attributes()
    with open("/home/felix/new_bachelor/awa2/dataset_analysis/class_occurrences.pickle", "rb") as act:
        class_cnt = pickle.load(act)
    act.close()
    class_idx_attribute = attrPred.class_idx_attribute
    categories = list(attr_converter_class.attribute_dict.keys())
    print(categories)
    for category in categories:
        category_idx_attribute = attr_converter_class.keep_category(
            category, copy.deepcopy(attrPred.idx_attribute))
        category_attrs = list(category_idx_attribute.values())
        result = analyzer.check_if_class_has_attribute(category_attrs, class_idx_attribute, class_cnt)
        print("Category: ", category, " Result: ", result)


def check_class_occurrences():
    """Show which class occures how often."""

    start = time.time()
    attrPred = AttributePredictor()
    res = attrPred.count_class_samples()


def experiment4(folder="", opposite=False):
    """Predict all the attributes one by one.
    Problem: A lot of true negatives, which distort the true performance. Therefore use recall as measurement."""

    start = time.time()
    attrPred = AttributePredictor()
    analyzer = Analyzer()
    attr_converter_class = Attributes()
    if opposite:
        raw_text = attrPred.create_text_labels_each(
            attrPred.idx_attribute, attr_converter_class.attribute_to_tuple_opposite)
    else:
        folder = "opposite4_attr_"
        raw_text = attrPred.create_text_labels_each(
            attrPred.idx_attribute, attr_converter_class.attribute_to_tuple)

    # pos_attr_mapping: Maps from the position of the attribtue in the text_features (0, ...) to the index of the attribute.
    text_features = attrPred.tokenize_text(raw_text, tuple_embedding=True)
    print("Text tokenized")
    attrPred.validate4(text_features, folder=folder)

    # Plot the results.
    analyzer.experiment_analysis_4(folder=folder, attr_idx_label_mapping=attrPred.idx_attribute, topx=20)
    end = time.time()
    print("Time consumption: ", end - start)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    # experiment1(simple_embedding=False, folder="improved_embedding_")
    # experiment2(simple_embedding=False, folder="improved_embedding_")
    # experiment_one_category3("shape_nouns", simple_embedding=False, folder="shape_nouns/")
    # check_attr_occurrence_in_class()
    # experiment_all_category3(simple_embedding=False)
    # check_attr_occurrence_in_class()
    # experiment_all_category3()
    experiment4()
    # analyze_shape_adj()
    # analyze_categories_relative(new_categories=True)
    # experiment1_per_attr_analysis()
