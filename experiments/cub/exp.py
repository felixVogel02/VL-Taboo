import copy
import pickle
import time
from cgi import test

import matplotlib.pyplot as plt
import torch
from load_dataset.cub.load_dataset import LoadDataset
from sentenceCreator.createSentences_CUB import SentenceCreator


class Experiment():
    """Execute the first experiment."""

    def __init__(self, model, device):
        """Initialize the class."""

        self.model = model
        self.device = device

    def validation1(self, data_loader, sentenceCreator, text_features, no_label=False, attr_num=1, file_path_base: str = "/home/felix/new_bachelor/cub/results/attr_vs_no_attr/"):
        """Predict the attributes for every image. To the label class is added one attribute. to the other classes not."""

        # Inner dict is for recall. If class i is the label, was it also predicted. (Not other way round!)
        result = {**{i: {"correct": 0, "count": 0} for i in range(1, 201)},
                  **{"total": {"correct": 0, "count": 0, "ignored_imgs": 0}}}
        cntEND = 0
        for input in data_loader:
            cntEND += 1
            if cntEND < 1000:
                continue
            use_text_feat = copy.deepcopy(text_features)
            img = input["image"]
            attributes = {key: input["attributes"][key][0] for key in input["attributes"].keys()}
            label = input["label_name"][0]
            class_id = input["class_id"][0]
            # Not enough attributes are surely assigned to this image. Ignore this image.
            if len(attributes.keys()) < attr_num:
                result["total"]["ignored_imgs"] = result["total"]["ignored_imgs"] + 1
                continue
            # Create new sentence for label class and replace the old sentence of that class in text_features.
            sent = sentenceCreator.create_class_sentences_with_attributes(
                label, attributes, attr_num=attr_num, no_label=no_label)
            tokenized_sent = sentenceCreator.tokenize_single_sentence(sent)
            # print("Sentence: ", sent)
            # print("Tokenized text: ", tokenized_sent)
            sent_idx = int(class_id) - 1  # Because indices in lists start with 0 not 1.
            use_text_feat[sent_idx] = tokenized_sent
            use_text_feat = sentenceCreator.norm_text(use_text_feat)

            with torch.no_grad():
                image_features = self.model.encode_image(img.to(self.device))

            image_features /= image_features.norm(dim=-1, keepdim=True)
            # similarity = (100.0 * image_features @ use_text_feat.T).softmax(dim=-1)
            sim = image_features @ use_text_feat.T
            similarity = 100 * sim
            torch.set_printoptions(precision=10)
            similarity = similarity.softmax(dim=-1)
            # print("Sim: ", sim)
            print("Similarity: ", similarity[0])
            values, indices = torch.topk(similarity[0], 1)
            cls_id_pred = indices[0].item() + 1
            label_cls = int(class_id)

            correct = False
            if cls_id_pred == label_cls:
                correct = True
            result[label_cls]["correct"] = result[label_cls]["correct"] + correct
            result[label_cls]["count"] = result[label_cls]["count"] + 1
            result["total"]["correct"] = result["total"]["correct"] + correct
            result["total"]["count"] = result["total"]["count"] + 1

        file_path = file_path_base + str(attr_num) + "_attr_results.pickle"
        with open(file_path, "wb") as act:
            pickle.dump(result, act, protocol=pickle.HIGHEST_PROTOCOL)
        act.close()

        return result

    def validation2(self, data_loader, sentenceCreator, id_label, attr_num=1, file_path_base: str = "/home/felix/new_bachelor/cub/results/all_same_attr/"):
        """Predict the attributes for every image. To all class sentences is added the attribute of the label class.
        Without an extra subfolder: Minimum certainty of attribtues is 3 ("probably")."""

        # Inner dict is for recall. If class i is the label, was it also predicted. (Not other way round!)
        result = {**{i: {"correct": 0, "count": 0} for i in range(1, 201)},
                  **{"total": {"correct": 0, "count": 0, "ignored_imgs": 0}}}
        for input in data_loader:
            img = input["image"]
            attributes = {key: input["attributes"][key][0] for key in input["attributes"].keys()}
            label = input["label_name"][0]
            class_id = input["class_id"][0]
            # Not enough attributes are surely assigned to this image. Ignore this image.
            if len(attributes.keys()) < attr_num:
                result["total"]["ignored_imgs"] = result["total"]["ignored_imgs"] + 1
                continue
            # Create new sentence for label class and replace the old sentence of that class in text_features.
            id_sentence = sentenceCreator.create_class_sentences_with_same_attr(
                id_label, attributes=attributes, attr_num=attr_num)
            text_features = sentenceCreator.tokenize_dict_text(text=id_sentence)
            text_features = sentenceCreator.norm_text(text_features)

            with torch.no_grad():
                image_features = self.model.encode_image(img.to(self.device))

            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            values, indices = similarity[0].topk(1)
            # print("Vlaues: ", values)
            # print("Indices: ", indices)
            # print("Label: ", label)
            # print("class id: ", class_id)
            cls_id_pred = indices[0].item() + 1
            label_cls = int(class_id)

            correct = False
            if cls_id_pred == label_cls:
                correct = True
            result[label_cls]["correct"] = result[label_cls]["correct"] + correct
            result[label_cls]["count"] = result[label_cls]["count"] + 1
            result["total"]["correct"] = result["total"]["correct"] + correct
            result["total"]["count"] = result["total"]["count"] + 1

        file_path = file_path_base + str(attr_num) + "_attr_results.pickle"
        with open(file_path, "wb") as act:
            pickle.dump(result, act, protocol=pickle.HIGHEST_PROTOCOL)
        act.close()

        return result


def experiment1(file_path_base, module, model_name="open_clip", name_add=""):
    """Executes the first experiment where attributes are added only to the correct sentence."""

    start = time.time()
    loadDataset = LoadDataset()
    infos = loadDataset.do_all(model_name=model_name)
    end1 = time.time()
    print("Loaded Dataset and model in: ", end1-start, "s")
    sentenceCreator = SentenceCreator(model=infos["model"], module=module, device="cuda")

    # Create the sentences and tokenize them.
    id_sentence = sentenceCreator.create_class_sentences(infos["class_names"])
    text_features = sentenceCreator.tokenize_dict_text(text=id_sentence)

    experiment = Experiment(model=infos["model"], device="cuda")
    for attr_num in range(0, 26):

        experiment.validation1(infos["data_loader"], sentenceCreator, text_features=text_features, attr_num=attr_num,
                               file_path_base=file_path_base)
        print("Done with attribute number: ", str(attr_num))
    end2 = time.time()
    print("Total time consumption: ", end2-start, "s")


def experiment2(file_path_base, module, model_name="open_clip", name_add=""):
    """Executes the second experiment where the attributes of the current image are added to all sentences."""

    start = time.time()
    loadDataset = LoadDataset()
    infos = loadDataset.do_all(model_name=model_name)
    end1 = time.time()
    print("Loaded Dataset and model in: ", end1-start, "s")
    sentenceCreator = SentenceCreator(model=infos["model"], module=module, device="cuda")

    experiment = Experiment(model=infos["model"], device="cuda")
    for attr_num in range(0, 25):
        experiment.validation2(infos["data_loader"], sentenceCreator, id_label=infos["class_names"], attr_num=attr_num,
                               file_path_base=file_path_base)
        print("Done with attribute number: ", str(attr_num))
    end2 = time.time()
    print("Total time consumption: ", end2-start, "s")


def experiment3(file_path_base, module, model_name="open_clip", name_add=""):
    """Executes experiment 3 where one image gets its image attributes, while all
    other images get the same number of attributes, but their class attributes.
    This is relative similar to experiment one, but now the initially created sentences are different."""

    start = time.time()
    loadDataset = LoadDataset()
    # Average of attr percentage is ca. 12%-23%.
    # avg_factor=1.5 -> Every class has at least 22 attributes.
    # avg_factor=2 -> Every class has at least 14 attributes.
    infos = loadDataset.do_all(avg_factor=1.5, model_name=model_name)
    end1 = time.time()
    print("Loaded Dataset and model in: ", end1-start, "s")
    sentenceCreator = SentenceCreator(model=infos["model"], module=module, device="cuda")
    mini = float("inf")

    # Create the sentences and tokenize them.
    experiment = Experiment(model=infos["model"], device="cuda")
    for attr_num in range(1, 26):
        # print(infos["id_attr"])
        id_sentence = sentenceCreator.create_class_sentences_with_different_attr(
            infos["class_names"], infos["class_attributes"], infos["id_attr"], attr_num)
        text_features = sentenceCreator.tokenize_dict_text(text=id_sentence)
        # print("Start: ", id_sentence[1])
        # print(id_sentence[2])
        print("Tokenized sentence: ", text_features)

        experiment.validation1(infos["data_loader"], sentenceCreator, text_features=text_features, attr_num=attr_num,
                               file_path_base=file_path_base)
        print("Done with attribute number: ", str(attr_num))
    end2 = time.time()
    print("Total time consumption: ", end2-start, "s")


def experiment4(file_path_base, module, model_name="open_clip", name_add=""):
    """Similar to experiment 3, but now the class labels are left out and only attributes are used."""

    start = time.time()
    loadDataset = LoadDataset()
    # Average of attr percentage is ca. 12%-23%.
    # avg_factor=1.5 -> Every class has at least 22 attributes.
    # avg_factor=2 -> Every class has at least 14 attributes.
    infos = loadDataset.do_all(avg_factor=1.5, model_name=model_name)
    end1 = time.time()
    print("Loaded Dataset and model in: ", end1-start, "s")
    sentenceCreator = SentenceCreator(model=infos["model"], module=module, device="cuda")

    # Create the sentences and tokenize them.
    experiment = Experiment(model=infos["model"], device="cuda")
    for attr_num in range(1, 26):
        # print(infos["id_attr"])
        id_sentence = sentenceCreator.create_class_sentences_with_different_attr(
            infos["class_names"], infos["class_attributes"], infos["id_attr"], attr_num, no_label=True)
        print("Id sentence: ", id_sentence)
        text_features = sentenceCreator.tokenize_dict_text(text=id_sentence)
        # continue
        print("Text features: ", text_features)

        experiment.validation1(infos["data_loader"], sentenceCreator, text_features=text_features, no_label=True, attr_num=attr_num,
                               file_path_base=file_path_base)
        print("Done with attribute number: ", str(attr_num))
    end2 = time.time()
    print("Total time consumption: ", end2-start, "s")


if __name__ == "__main__":
    torch.set_printoptions(precision=10)
    # experiment1(module=clip, model_name="clip", name_add="clip")
