import sys

sys.path.append("/home/felix/new_bachelor/cub/")  # noqa


import copy
import pickle
import random
import time
from cgi import test

import numpy as np
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
        for input in data_loader:

            use_text_feat = copy.deepcopy(text_features)
            image_features = input["image"].to(self.device)
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
            print("Sentence: ", sent)
            # print("Tokenized text: ", tokenized_sent)
            sent_idx = int(class_id) - 1  # Because indices in lists start with 0 not 1.
            use_text_feat[sent_idx] = tokenized_sent
            use_text_feat = sentenceCreator.norm_text(use_text_feat).to(self.device)

            sim = image_features @ use_text_feat.T
            similarity = 100 * sim
            similarity = similarity.softmax(dim=-1)

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
        print("Accuracy: ", result["total"]["correct"] / result["total"]["count"])
        with open(file_path, "wb") as act:
            pickle.dump(result, act, protocol=pickle.HIGHEST_PROTOCOL)
        act.close()

        return result

    def validation1_stat(self, data_loader, sentenceCreator, infos, no_label=False, attr_num=1, file_path_base: str = "/home/felix/new_bachelor/cub/results/attr_vs_no_attr/"):
        """Predict the attributes for every image. To the label class is added one attribute. to the other classes not."""

        # absolute count: List with an entry per image. Value is the number of prompts that fit the image attributes (including the image prompt).
        # total_count because sometimes images are ignored when they have to few attributes.
        result = {"absolute_count": []}
        for input in data_loader:
            attributes = {key: input["attributes"][key][0] for key in input["attributes"].keys()}
            label = input["label_name"][0]
            class_id = input["class_id"][0]

            # Not enough attributes are surely assigned to this image. Ignore this image.
            if len(attributes.keys()) < attr_num:
                continue

            # Create class sentences.
            all_attr = sentenceCreator.create_class_sentences_with_different_attr(
                infos["class_names"], infos["class_attributes"], infos["id_attr"], attr_num, no_label=True, only_chosed_attr=True, ignore_label=label)

            id_attr = infos["id_attr"]
            same = 0
            for attrs in all_attr:
                # For this class attrs has been chosen as the attributes. Check if all of them are present in the
                # image attributes of the current image.
                doubled = True  # True if it has only attributes of the image.
                for attr in attrs:
                    descr = id_attr[str(attr)][0]
                    val = id_attr[str(attr)][1]
                    if descr in attributes.keys():
                        # print("Value: ", attributes[descr][0])  # 0 because it is a lsit with only one element.
                        if val != attributes[descr][0]:
                            doubled = False
                            break  # This attribute is not present in the image.
                    else:  # At least one attribute does not fit the image attributes.
                        doubled = False
                        break
                same += doubled

            result["absolute_count"].append(same+1)

        file_path = file_path_base + str(attr_num) + "statistic_same_sentence.pickle"
        with open(file_path, "wb") as act:
            pickle.dump(result, act, protocol=pickle.HIGHEST_PROTOCOL)
        act.close()

        return result

    def validation1_new(self, data_loader, sentenceCreator, text_list, no_label=False, attr_num=1, file_path_base: str = "/home/felix/new_bachelor/cub/results/attr_vs_no_attr/"):
        """Predict the attributes for every image. To the label class is added one attribute. to the other classes not."""

        # Inner dict is for recall. If class i is the label, was it also predicted. (Not other way round!)
        result = {**{i: {"correct": 0, "count": 0} for i in range(1, 201)},
                  **{"total": {"correct": 0, "count": 0, "ignored_imgs": 0}}}
        for input in data_loader:

            # First choose text features from the sentences with different randomly chosen attributes.
            use_text_feat = torch.zeros(np.shape(text_list[0])[0], np.shape(text_list[0])[1])
            # print("Shape use_text_feat:", np.shape(use_text_feat))
            for i in range(np.shape(text_list[0])[0]):
                # Choose each sentence independently.
                full_text = random.choice(text_list)
                use_text_feat[i] = full_text[i]

            # use_text_feat = copy.deepcopy(text_features)
            image_features = input["image"].to(self.device)
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
            if label == "Lazuli Bunting" or label == "Pelagic Cormorant":
                print("Label: ", label, "Sentence: ", sent)
            # print("Tokenized text: ", tokenized_sent)
            sent_idx = int(class_id) - 1  # Because indices in lists start with 0 not 1.
            use_text_feat[sent_idx] = tokenized_sent
            use_text_feat = sentenceCreator.norm_text(use_text_feat).to(self.device)

            sim = image_features @ use_text_feat.T
            similarity = 100 * sim
            similarity = similarity.softmax(dim=-1)

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
        print("Accuracy: ", result["total"]["correct"] / result["total"]["count"])
        # with open(file_path, "wb") as act:
        #     pickle.dump(result, act, protocol=pickle.HIGHEST_PROTOCOL)
        # act.close()

        return result

    def validation2(self, data_loader, sentenceCreator, id_label, attr_num=1, file_path_base: str = "/home/felix/new_bachelor/cub/results/all_same_attr/"):
        """Predict the attributes for every image. To all class sentences is added the attribute of the label class.
        Without an extra subfolder: Minimum certainty of attribtues is 3 ("probably")."""

        # Inner dict is for recall. If class i is the label, was it also predicted. (Not other way round!)
        result = {**{i: {"correct": 0, "count": 0} for i in range(1, 201)},
                  **{"total": {"correct": 0, "count": 0, "ignored_imgs": 0}}}
        doit = True
        for input in data_loader:
            image_features = input["image"].to(self.device)
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
            if doit and "Black footed Albatross" in label:
                doit = False
                print(id_sentence[1], "+++", id_sentence[2], "+++", id_sentence[10])
            elif "Crested" in label:
                print(label)
                print(id_sentence[int(class_id)], "+++", id_sentence[2], "+++", id_sentence[10])
                return

            text_features = sentenceCreator.tokenize_dict_text(text=id_sentence)
            text_features = sentenceCreator.norm_text(text_features).to(self.device)

            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            values, indices = similarity[0].topk(1)
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
        print("Accuracy: ", result["total"]["correct"] / result["total"]["count"])
        with open(file_path, "wb") as act:
            pickle.dump(result, act, protocol=pickle.HIGHEST_PROTOCOL)
        act.close()

        return result

    def validation3(self, data_loader, sentenceCreator, infos, all_attributes=[], all_attributes_search=[], all_attributes_search_inv=[], no_label=False, attr_num=1, file_path_base: str = "/home/felix/new_bachelor/cub/results/attr_vs_no_attr/"):
        """Predict the attributes for every image. To the label class is added one attribute. to the other classes not."""

        # Inner dict is for recall. If class i is the label, was it also predicted. (Not other way round!)
        result = {**{i: {"correct": 0, "count": 0} for i in range(1, 201)},
                  **{"total": {"correct": 0, "count": 0, "ignored_imgs": 0}}}
        for input in data_loader:

            image_features = input["image"].type(torch.DoubleTensor).to(self.device)
            attributes = {key: input["attributes"][key][0] for key in input["attributes"].keys()}
            label = input["label_name"][0]
            class_id = input["class_id"][0]

            # Not enough attributes are surely assigned to this image. Ignore this image.
            if len(attributes.keys()) < attr_num:
                result["total"]["ignored_imgs"] = result["total"]["ignored_imgs"] + 1
                continue

            # Create "attributes" for wrong attributes that don't fit the image.
            # print("Attributes: ", attributes)
            check_attributes = []
            for key in attributes.keys():
                check_attributes.append(key + " " + attributes[key][0])

            # Delete image attributes.
            all_attributes_loc = copy.deepcopy(all_attributes)
            rem_idx = []
            for attr in check_attributes:
                idx = all_attributes_search_inv[attr]
                del all_attributes_loc[idx]
                rem_idx.append(idx)

            # List the wrong attributes in the correct form.
            wrong_attributes = dict()
            for key in all_attributes_loc.keys():
                descr = all_attributes_loc[key][0]
                val = all_attributes_loc[key][1]
                wrong_attributes[descr] = [val]

            # Choose random wrong class name:
            # print(infos["class_names"])
            poss_class_names = copy.deepcopy(list(infos["class_names"].values()))
            poss_class_names.remove(label)
            wrong_label = random.choice(poss_class_names)

            # Create sentence with correct label and wrong attributes:
            sent = sentenceCreator.create_class_sentences_with_attributes(
                label, wrong_attributes, attr_num=attr_num, no_label=no_label)
            tokenized_sent = sentenceCreator.tokenize_single_sentence(sent)

            # Create sentence with wrong label and correct attributes.
            sent_wrong = sentenceCreator.create_class_sentences_with_attributes(
                wrong_label, attributes, attr_num=attr_num, no_label=no_label)
            tokenized_sent_wrong = sentenceCreator.tokenize_single_sentence(sent_wrong)

            use_text_feat = torch.cat([tokenized_sent, tokenized_sent_wrong]).type(torch.DoubleTensor)
            # print("Use text feat: ", np.shape(use_text_feat))

            # Random attributes that are not in "attributes":

            use_text_feat = sentenceCreator.norm_text(use_text_feat).to(self.device)

            sim = image_features @ use_text_feat.T
            similarity = 100 * sim
            similarity = similarity.softmax(dim=-1)

            values, indices = torch.topk(similarity[0], 1)

            cls_id_pred = indices[0].item() + 1
            label_cls = int(class_id)

            correct = False
            if indices[0].item() == 0:  # The correct sentence (with the correct label) is alwys put at position 0.
                correct = True
            result[label_cls]["correct"] = result[label_cls]["correct"] + correct
            result[label_cls]["count"] = result[label_cls]["count"] + 1
            result["total"]["correct"] = result["total"]["correct"] + correct
            result["total"]["count"] = result["total"]["count"] + 1

        file_path = file_path_base + str(attr_num) + "one_vs_one.pickle"
        print("Accuracy: ", result["total"]["correct"] / result["total"]["count"])
        with open(file_path, "wb") as act:
            pickle.dump(result, act, protocol=pickle.HIGHEST_PROTOCOL)
        act.close()

        return result


def experiment1(file_path_base, module, model_name="open_clip", name_add=""):
    """Executes the first experiment where attribtues are added only to the correct sentence.
    module: Not needed for flava model. (Internally processor is used instead.)"""

    start = time.time()
    loadDataset = LoadDataset()
    infos = loadDataset.do_all_new(model_name=model_name, batch_size=1)
    end1 = time.time()
    print("Loaded Dataset and model in: ", end1-start, "s")
    sentenceCreator = SentenceCreator(model=infos["model"], module=module,
                                      device="cuda", processor=infos["processor"], model_name=model_name)
    # Create the sentences and tokenize them.
    id_sentence = sentenceCreator.create_class_sentences(infos["class_names"])
    # return
    text_features = sentenceCreator.tokenize_dict_text(text=id_sentence)

    experiment = Experiment(model=infos["model"], device="cuda")
    for attr_num in [2]:  # range(0, 20):
        start1 = time.time()
        print(id_sentence[1], "+++++", id_sentence[2], "+++++", id_sentence[3], "+++++")

        experiment.validation1(infos["data_loader"], sentenceCreator, text_features=text_features, attr_num=attr_num,
                               file_path_base=file_path_base)
        print("Done with attribute number: ", str(attr_num))
        end1 = time.time()
        print(f"Step {attr_num} took {end1-start1} s.")
    end2 = time.time()
    print("Total time consumption: ", end2-start, "s")


def experiment2(file_path_base, module, model_name="open_clip", name_add=""):
    """Executes the second experiment where the attributes of the current image are added to all sentences."""

    start = time.time()
    loadDataset = LoadDataset()
    infos = loadDataset.do_all_new(model_name=model_name, batch_size=1)
    end1 = time.time()
    print("Loaded Dataset and model in: ", end1-start, "s")
    sentenceCreator = SentenceCreator(model=infos["model"], module=module,
                                      device="cuda", processor=infos["processor"], model_name=model_name)

    experiment = Experiment(model=infos["model"], device="cuda")
    for attr_num in [2]:  # range(0, 20):
        start1 = time.time()
        experiment.validation2(infos["data_loader"], sentenceCreator, id_label=infos["class_names"], attr_num=attr_num,
                               file_path_base=file_path_base)
        end1 = time.time()
        print(f"Step {attr_num} took {end1-start1} s.")
    end2 = time.time()
    print("Total time consumption: ", end2-start, "s")


def experiment3(file_path_base, module, model_name="open_clip", name_add=""):
    """Executes experiment 3 where one image gets its image attributes, while all
    other images get the same number of attributes, but their class attributes.
    This is relative similar to experiment one, but now the initially created sentences are different."""

    start = time.time()
    loadDataset = LoadDataset()
    infos = loadDataset.do_all_new(model_name=model_name, batch_size=1, avg_factor=1.5)
    end1 = time.time()
    print("Loaded Dataset and model in: ", end1-start, "s")
    sentenceCreator = SentenceCreator(model=infos["model"], module=module,
                                      device="cuda", processor=infos["processor"], model_name=model_name)

    experiment = Experiment(model=infos["model"], device="cuda")

    for attr_num in [2]:  # range(0, 9):
        start1 = time.time()
        text_list = []  # 100 sentences with different randomly chosen attributes.
        for i in range(100):  # Create a selection of different sentences to choose from.
            id_sentence = sentenceCreator.create_class_sentences_with_different_attr(
                infos["class_names"], infos["class_attributes"], infos["id_attr"], attr_num)

            text_feat = sentenceCreator.tokenize_dict_text(text=id_sentence)
            text_list.append(text_feat)

        experiment.validation1_new(infos["data_loader"], sentenceCreator, text_list=text_list, attr_num=attr_num,
                                   file_path_base=file_path_base)
        print("Done with attribute number: ", str(attr_num))
        end1 = time.time()
        print(f"Step {attr_num} took {end1-start1} s.")
    end2 = time.time()
    print("Total time consumption: ", end2-start, "s")


def experiment4(file_path_base, module, model_name="open_clip", name_add=""):
    """Count how many classes the chosen attribute subset fits to get a statistical upper bound on the accuracy of class prediction."""

    start = time.time()
    loadDataset = LoadDataset()
    # Average of attr percentage is ca. 12%-23%.
    # avg_factor=1.5 -> Every class has at least 22 attributes.
    # avg_factor=2 -> Every class has at least 14 attributes.
    infos = loadDataset.do_all_new(model_name=model_name, batch_size=1, avg_factor=1.5)
    end1 = time.time()
    print("Loaded Dataset and model in: ", end1-start, "s")
    sentenceCreator = SentenceCreator(model=infos["model"], module=module,
                                      device="cuda", processor=infos["processor"], model_name=model_name)

    experiment = Experiment(model=infos["model"], device="cuda")
    for attr_num in [3]:  # range(0, 9):
        start1 = time.time()
        text_list = []  # 100 sentences with different randomly chosen attributes.
        for i in range(100):  # Create a selection of different sentences to choose from.
            id_sentence = sentenceCreator.create_class_sentences_with_different_attr(
                infos["class_names"], infos["class_attributes"], infos["id_attr"], attr_num, no_label=True)

            text_feat = sentenceCreator.tokenize_dict_text(text=id_sentence)
            text_list.append(text_feat)

        experiment.validation1_new(infos["data_loader"], sentenceCreator, text_list=text_list, no_label=True, attr_num=attr_num,
                                   file_path_base=file_path_base)

        end1 = time.time()
        print(f"Step {attr_num} took {end1-start1} s.")
    end2 = time.time()
    print("Total time consumption: ", end2-start, "s")


def experiment4_stat(file_path_base, module, model_name="open_clip", name_add=""):
    """It can be the case that several prompts look exactly the same, because they only use attributes that can duplicate.
        Even if the model would work perfectly it wouldn't know which of them to choose. Therefore we calculate approximately
        how many same """

    start = time.time()
    loadDataset = LoadDataset()
    # Average of attr percentage is ca. 12%-23%.
    # avg_factor=1.5 -> Every class has at least 22 attributes.
    # avg_factor=2 -> Every class has at least 14 attributes.
    infos = loadDataset.do_all_new(model_name=model_name, batch_size=1, avg_factor=1.5)
    end1 = time.time()
    print("Loaded Dataset and model in: ", end1-start, "s")
    sentenceCreator = SentenceCreator(model=infos["model"], module=module,
                                      device="cuda", processor=infos["processor"], model_name=model_name)

    experiment = Experiment(model=infos["model"], device="cuda")
    for attr_num in range(1, 9):
        start1 = time.time()

        experiment.validation1_stat(infos["data_loader"], sentenceCreator, infos, no_label=True, attr_num=attr_num,
                                    file_path_base=file_path_base)

        end1 = time.time()
        print(f"Step {attr_num} took {end1-start1} s.")
    end2 = time.time()
    print("Total time consumption: ", end2-start, "s")


def experiment5(file_path_base, module, model_name="open_clip", name_add=""):
    """Executes experiment 3 where one image gets its image attributes, while all
    other images get the same number of attributes, but their class attributes.
    This is relative similar to experiment one, but now the initially created sentences are different."""

    start = time.time()
    loadDataset = LoadDataset()
    infos = loadDataset.do_all_new(model_name=model_name, batch_size=1, avg_factor=1.5)
    end1 = time.time()
    print("Loaded Dataset and model in: ", end1-start, "s")
    sentenceCreator = SentenceCreator(model=infos["model"], module=module,
                                      device="cuda", processor=infos["processor"], model_name=model_name)

    experiment = Experiment(model=infos["model"], device="cuda")
    all_attributes = infos["id_attr"]
    all_attributes_search = copy.deepcopy(all_attributes)
    all_attributes_search_inv = dict()  # Same as all_attributes_search, but keys and values are exchanged.
    for key in all_attributes_search.keys():
        val = all_attributes_search[key][0] + " " + all_attributes_search[key][1]
        all_attributes_search[key] = val
        all_attributes_search_inv[val] = key

    for attr_num in range(1, 9):
        start1 = time.time()

        experiment.validation3(infos["data_loader"], sentenceCreator, infos, all_attributes=all_attributes, all_attributes_search=all_attributes_search,
                               all_attributes_search_inv=all_attributes_search_inv, attr_num=attr_num,
                               file_path_base=file_path_base)
        print("Done with attribute number: ", str(attr_num))
        end1 = time.time()
        print(f"Step {attr_num} took {end1-start1} s.")
    end2 = time.time()
    print("Total time consumption: ", end2-start, "s")


if __name__ == "__main__":
    torch.set_printoptions(precision=10)
    # experiment1(file_path_base, module="", model_name="flava", name_add="flava/")
