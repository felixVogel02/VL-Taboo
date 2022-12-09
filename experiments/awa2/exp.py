import copy
import pickle
import random
import time
from cgi import test

import numpy as np
import torch
from sentenceCreator.createSentences_AWA2 import SentenceCreator
from load_dataset.awa2.load_dataset import LoadDataset


class Experiment():
    """Execute the first experiment."""

    def __init__(self, model, device):
        """Initialize the class."""

        self.model = model
        self.device = device

    def validation1(self, data_loader, sentenceCreator, text_features, no_label=False, attr_num=1,
                    file_path_base: str = "/home/felix/new_bachelor/cub/results/attr_vs_no_attr/", adder="new_sent", complex_sent=False):
        """Predict the attributes for every image. To the label class is added one attribute. to the other classes not."""

        # Inner dict is for recall. If class i is the label, was it also predicted. (Not other way round!)
        result = {**{i: {"correct": 0, "count": 0} for i in range(1, 201)},
                  **{"total": {"correct": 0, "count": 0, "ignored_imgs": 0}}}
        cnt = 0
        print("Start validation: ")
        cnt = 0
        for input in data_loader:
            cnt += 1
            if cnt >= 5:
                return
            use_text_feat = copy.deepcopy(text_features)
            image_features = input["image"].type(torch.DoubleTensor).to(self.device)
            attributes = {key: input["img_attrs"][key][0] for key in input["img_attrs"].keys()}
            label = input["label_name"][0]
            class_id = input["class_id"][0]
            img_id = input["img_id"][0]
            image_name = input["image_name"][0]

            # Not enough attributes are surely assigned to this image. Ignore this image.
            if len(attributes.keys()) < attr_num:
                result["total"]["ignored_imgs"] = result["total"]["ignored_imgs"] + 1
                continue

            # Create new sentence for label class and replace the old sentence of that class in text_features.
            sent = sentenceCreator.create_class_sentences_with_attributes(
                label, attributes, attr_num=attr_num, no_label=no_label, complex_sent=complex_sent)
            tokenized_sent = sentenceCreator.tokenize_single_sentence(sent)
            # print("sent: ", sent)
            sent_idx = int(class_id) - 1  # Because indices in lists start with 0 not 1.
            use_text_feat[sent_idx] = tokenized_sent
            use_text_feat = sentenceCreator.norm_text(use_text_feat).type(torch.DoubleTensor).to(self.device)
            similarity = (100.0 * image_features @ use_text_feat.T).softmax(dim=-1)

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

        file_path = file_path_base + str(attr_num) + "_attr_results" + adder + ".pickle"
        with open(file_path, "wb") as act:  # Overwritten with new wrong data for experiment1 with OpenCLIP.
            pickle.dump(result, act, protocol=pickle.HIGHEST_PROTOCOL)
        act.close()

        return result

    def validation1_stat(self, data_loader, sentenceCreator, infos, no_label=False, attr_num=1, file_path_base: str = "/home/felix/new_bachelor/cub/results/attr_vs_no_attr/"):
        """Predict the attributes for every image. To the label class is added one attribute. to the other classes not."""

        # absolute count: List with an entry per image. Value is the number of prompts that fit the image attributes (including the image prompt).
        result = {"absolute_count": []}
        for input in data_loader:
            image_features = input["image"].type(torch.DoubleTensor).to(self.device)
            attributes = {key: input["img_attrs"][key][0] for key in input["img_attrs"].keys()}
            label = input["label_name"][0]
            class_id = input["class_id"][0]
            img_id = input["img_id"][0]
            image_name = input["image_name"][0]

            # Not enough attributes are surely assigned to this image. Ignore this image.
            if len(attributes.keys()) < attr_num:
                continue

            # Create class sentences.
            all_attr = sentenceCreator.create_class_sentences_with_different_attr(
                infos["class_names"], infos["class_attributes"], attr_num, no_label=True, only_chosed_attr=True, ignore_label=label)

            id_attr = infos["id_attr"]
            same = 0
            for attrs in all_attr:
                # For this class attrs has been chosen as the attributes. Check if all of them are present in the
                # image attributes of the current image.
                doubled = True  # True if it has only attributes of the image.
                for attr in attrs:
                    if attr not in attributes.values():
                        doubled = False
                        break  # This attribute is not present in the image.

                same += doubled
            result["absolute_count"].append(same+1)

        file_path = file_path_base + str(attr_num) + "statistic_same_sentence.pickle"
        with open(file_path, "wb") as act:
            pickle.dump(result, act, protocol=pickle.HIGHEST_PROTOCOL)
        act.close()

        return result

    def validation1_new(self, data_loader, sentenceCreator, text_list, no_label=False, attr_num=1,
                        file_path_base: str = "/home/felix/new_bachelor/cub/results/attr_vs_no_attr/", adder="", complex_sent=False):
        """Predict the attributes for every image. To the label class is added one attribute. to the other classes not."""

        # Inner dict is for recall. If class i is the label, was it also predicted. (Not other way round!)
        result = {**{i: {"correct": 0, "count": 0} for i in range(1, 201)},
                  **{"total": {"correct": 0, "count": 0, "ignored_imgs": 0}}}
        cnt = 0
        print("Start validation: ")
        for input in data_loader:

            use_text_feat = torch.zeros(np.shape(text_list[0])[0], np.shape(text_list[0])[1])
            # print("Shape use_text_feat:", np.shape(use_text_feat))
            for i in range(np.shape(text_list[0])[0]):
                # Choose each sentence independently.
                full_text = random.choice(text_list)
                use_text_feat[i] = full_text[i]

            image_features = input["image"].type(torch.DoubleTensor).to(self.device)
            attributes = {key: input["img_attrs"][key][0] for key in input["img_attrs"].keys()}
            label = input["label_name"][0]
            class_id = input["class_id"][0]
            # Not enough attributes are surely assigned to this image. Ignore this image.
            if len(attributes.keys()) < attr_num:
                result["total"]["ignored_imgs"] = result["total"]["ignored_imgs"] + 1
                continue

            # Create new sentence for label class and replace the old sentence of that class in text_features.
            sent = sentenceCreator.create_class_sentences_with_attributes(
                label, attributes, attr_num=attr_num, no_label=no_label, complex_sent=complex_sent)
            tokenized_sent = sentenceCreator.tokenize_single_sentence(sent)
            use_text_feat[sent_idx] = tokenized_sent
            use_text_feat = sentenceCreator.norm_text(use_text_feat).type(torch.DoubleTensor).to(self.device)
            similarity = (100.0 * image_features @ use_text_feat.T).softmax(dim=-1)

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

        file_path = file_path_base + str(attr_num) + "_attr_results" + adder + ".pickle"
        with open(file_path, "wb") as act:
            pickle.dump(result, act, protocol=pickle.HIGHEST_PROTOCOL)
        act.close()

        return result

    def validation2(self, data_loader, sentenceCreator, id_label, attr_num=1,
                    file_path_base: str = "/home/felix/new_bachelor/cub/results/all_same_attr/", adder="new_sent", complex_sent=False):
        """Predict the attributes for every image. To all class sentences is added the attribute of the label class.
        Without an extra subfolder: Minimum certainty of attribtues is 3 ("probably")."""

        # Inner dict is for recall. If class i is the label, was it also predicted. (Not other way round!)
        result = {**{i: {"correct": 0, "count": 0} for i in range(1, 201)},
                  **{"total": {"correct": 0, "count": 0, "ignored_imgs": 0}}}
        cnt = 0
        doit = True
        for input in data_loader:
            cnt += 1
            image_features = input["image"].type(torch.DoubleTensor).to(self.device)
            attributes = {key: input["img_attrs"][key][0] for key in input["img_attrs"].keys()}
            label = input["label_name"][0]
            class_id = input["class_id"][0]
            # Not enough attributes are surely assigned to this image. Ignore this image.
            if len(attributes.keys()) < attr_num:
                result["total"]["ignored_imgs"] = result["total"]["ignored_imgs"] + 1
                continue
            # Create new sentence for label class and replace the old sentence of that class in text_features.
            id_sentence = sentenceCreator.create_class_sentences_with_same_attr(
                id_label, attributes=attributes, attr_num=attr_num, complex_sent=complex_sent)
            text_features = sentenceCreator.tokenize_dict_text(text=id_sentence)
            text_features = sentenceCreator.norm_text(text_features).type(torch.DoubleTensor).to(self.device)

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

        file_path = file_path_base + str(attr_num) + "_attr_results" + adder + ".pickle"
        with open(file_path, "wb") as act:
            pickle.dump(result, act, protocol=pickle.HIGHEST_PROTOCOL)
        act.close()

        return result

    def validation3(self, data_loader, sentenceCreator, class_names, all_attributes=[], all_attributes_search=[], all_attributes_search_inv=[],
                    no_label=False, attr_num=1, file_path_base: str = "/home/felix/new_bachelor/cub/results/attr_vs_no_attr/", adder="new_sent", complex_sent=True):
        """Predict the attributes for every image. To the label class is added one attribute. to the other classes not."""

        # Inner dict is for recall. If class i is the label, was it also predicted. (Not other way round!)
        result = {**{i: {"correct": 0, "count": 0} for i in range(1, 201)},
                  **{"total": {"correct": 0, "count": 0, "ignored_imgs": 0}}}
        for input in data_loader:

            image_features = input["image"].type(torch.DoubleTensor).to(self.device)
            attributes = {key: input["img_attrs"][key][0] for key in input["img_attrs"].keys()}
            label = input["label_name"][0]
            class_id = input["class_id"][0]
            img_id = input["img_id"][0]
            image_name = input["image_name"][0]

            # Not enough attributes are surely assigned to this image. Ignore this image.
            if len(attributes.keys()) < attr_num:
                result["total"]["ignored_imgs"] = result["total"]["ignored_imgs"] + 1
                continue

            wrong_attributes = copy.deepcopy(all_attributes)
            #print("Wrong attributes: ", len(wrong_attributes))
            for attr in attributes.values():
                for key in all_attributes.keys():
                    if key in wrong_attributes.keys() and wrong_attributes[key] == attr:
                        del wrong_attributes[key]

            all_labels = copy.deepcopy(list(class_names.values()))
            all_labels.remove(label)
            wrong_label = random.choice(all_labels)

            # Create sentence with correct label and wrong attributes:
            sent = sentenceCreator.create_class_sentences_with_attributes(
                label, wrong_attributes, attr_num=attr_num, no_label=no_label, complex_sent=complex_sent)
            tokenized_sent = sentenceCreator.tokenize_single_sentence(sent)

            # Create sentence with wrong label and correct attributes.
            sent_wrong = sentenceCreator.create_class_sentences_with_attributes(
                wrong_label, attributes, attr_num=attr_num, no_label=no_label, complex_sent=complex_sent)
            tokenized_sent_wrong = sentenceCreator.tokenize_single_sentence(sent_wrong)

            use_text_feat = torch.cat([tokenized_sent, tokenized_sent_wrong]).type(torch.DoubleTensor)

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
            # print(correct)

        file_path = file_path_base + str(attr_num) + "one_vs_one" + adder + ".pickle"
        print("Accuracy: ", result["total"]["correct"] / result["total"]["count"])
        with open(file_path, "wb") as act:
            pickle.dump(result, act, protocol=pickle.HIGHEST_PROTOCOL)
        act.close()

        return result


def experiment1(module, model_name="open_clip", name_add=""):
    """Executes the first experiment where attributes are added only to the correct sentence."""

    start = time.time()
    loadDataset = LoadDataset(model_name=model_name, new=True)
    model = loadDataset.model
    class_names = loadDataset.idx_label
    dataloader = loadDataset.data_loader
    processor = loadDataset.processor
    class_idx_attribute = loadDataset.class_idx_attribute
    end1 = time.time()
    print("Loaded Dataset and model in: ", end1-start, "s")
    sentenceCreator = SentenceCreator(model=model, module=module, device="cuda",
                                      processor=processor, model_name=model_name)

    # Create the sentences and tokenize them.
    id_sentence = sentenceCreator.create_class_sentences(class_names)
    # return
    text_features = sentenceCreator.tokenize_dict_text(text=id_sentence)
    # print("Id sentence: ", id_sentence)

    experiment = Experiment(model, device="cuda")
    for attr_num in [3]:  # range(0, 21):
        start1 = time.time()
        # print("ID sentence: ", id_sentence[1], "++++", id_sentence[2])
        experiment.validation1(dataloader, sentenceCreator, text_features=text_features, attr_num=attr_num,
                               file_path_base="/home/felix/new_bachelor/awa2/results/" + name_add + "attr_vs_no_attr/")
        end1 = time.time()
        print("Done with attribute number: ", str(attr_num), "in: ", end1-start1, "s")
    end2 = time.time()
    print("Total time consumption: ", end2-start, "s")


def experiment2(module, model_name="open_clip", name_add="", starter=0):
    """Executes the second experiment where the attributes of the current image are added to all sentences."""

    start = time.time()
    loadDataset = LoadDataset(model_name=model_name, new=True)
    model = loadDataset.model
    class_names = loadDataset.idx_label
    dataloader = loadDataset.data_loader
    processor = loadDataset.processor
    end1 = time.time()
    print("Loaded Dataset and model in: ", end1-start, "s")
    sentenceCreator = SentenceCreator(model=model, module=module, device="cuda",
                                      processor=processor, model_name=model_name)

    experiment = Experiment(model, device="cuda")
    for attr_num in [2]:  # range(starter, 21):
        start1 = time.time()
        # data_loader, sentenceCreator, id_label, attr_num=1, file_path_base
        experiment.validation2(dataloader, sentenceCreator, id_label=class_names, attr_num=attr_num,
                               file_path_base="/home/felix/new_bachelor/awa2/results/" + name_add + "all_same_attr/")
        end1 = time.time()
        print("Done with attribute number: ", str(attr_num), "in: ", end1-start1, "s")
    end2 = time.time()
    print("Total time consumption: ", end2-start, "s")


def experiment3(module, model_name="open_clip", name_add=""):
    """Executes experiment 3 where one image gets its image attributes, while all
    other images get the same number of attributes, but their class attributes.
    This is relative similar to experiment one, but now the initially created sentences are different."""

    start = time.time()
    loadDataset = LoadDataset(model_name=model_name, new=True)
    model = loadDataset.model
    class_names = loadDataset.idx_label
    dataloader = loadDataset.data_loader
    class_idx_attribute = loadDataset.class_idx_attribute
    processor = loadDataset.processor
    # print(class_idx_attribute)
    end1 = time.time()
    print("Loaded Dataset and model in: ", end1-start, "s")
    sentenceCreator = SentenceCreator(model=model, module=module, device="cuda",
                                      processor=processor, model_name=model_name)
    # class_idx_attribute

    experiment = Experiment(model, device="cuda")
    for attr_num in [2]:  # range(0, 21):
        start1 = time.time()

        text_list = []  # 100 sentences with different randomly chosen attributes.
        for i in range(100):  # Create a selection of different sentences to choose from.
            id_sentence = sentenceCreator.create_class_sentences_with_different_attr(
                class_names, class_idx_attribute, attr_num)
            print(id_sentence[3], "++++", id_sentence[20], "++++", id_sentence[18], "++++", id_sentence[31], "++++")
            text_feat = sentenceCreator.tokenize_dict_text(text=id_sentence)
            text_list.append(text_feat)

        experiment.validation1_new(dataloader, sentenceCreator, text_list=text_list, attr_num=attr_num,
                                   file_path_base="/home/felix/new_bachelor/awa2/results/" + name_add + "attr_vs_class_attr/")
        end1 = time.time()
        print("Done with attribute number: ", str(attr_num), "in: ", end1-start1, "s")
    end2 = time.time()
    print("Total time consumption: ", end2-start, "s")


def experiment1_new_sent(module, model_name="open_clip", name_add=""):
    """Executes the first experiment where attributes are added only to the correct sentence."""

    start = time.time()
    loadDataset = LoadDataset(model_name=model_name, new=True)
    model = loadDataset.model
    class_names = loadDataset.idx_label
    dataloader = loadDataset.data_loader
    processor = loadDataset.processor
    class_idx_attribute = loadDataset.class_idx_attribute
    end1 = time.time()
    print("Loaded Dataset and model in: ", end1-start, "s")
    sentenceCreator = SentenceCreator(model=model, module=module, device="cuda",
                                      processor=processor, model_name=model_name)

    # Create the sentences and tokenize them.
    id_sentence = sentenceCreator.create_class_sentences(class_names)
    # return
    text_features = sentenceCreator.tokenize_dict_text(text=id_sentence)
    # print("Id sentence: ", id_sentence)

    experiment = Experiment(model, device="cuda")
    for attr_num in range(0, 21):
        start1 = time.time()
        # print("ID sentence: ", id_sentence[1], "++++", id_sentence[2])
        experiment.validation1(dataloader, sentenceCreator, text_features=text_features, attr_num=attr_num,
                               file_path_base="/home/felix/new_bachelor/awa2/results/" + name_add + "attr_vs_no_attr/",
                               adder="new_sent", complex_sent=True)
        end1 = time.time()
        print("Experiment1: Done with attribute number: ", str(attr_num), "in: ", end1-start1, "s")
    end2 = time.time()
    print("Total time consumption: ", end2-start, "s")


def experiment2_new_sent(module, model_name="open_clip", name_add="", starter=0):
    """Executes the second experiment where the attributes of the current image are added to all sentences."""

    start = time.time()
    loadDataset = LoadDataset(model_name=model_name, new=True)
    model = loadDataset.model
    class_names = loadDataset.idx_label
    dataloader = loadDataset.data_loader
    processor = loadDataset.processor
    end1 = time.time()
    print("Loaded Dataset and model in: ", end1-start, "s")
    sentenceCreator = SentenceCreator(model=model, module=module, device="cuda",
                                      processor=processor, model_name=model_name)

    experiment = Experiment(model, device="cuda")
    for attr_num in range(starter, 21):
        start1 = time.time()
        # data_loader, sentenceCreator, id_label, attr_num=1, file_path_base
        experiment.validation2(dataloader, sentenceCreator, id_label=class_names, attr_num=attr_num,
                               file_path_base="/home/felix/new_bachelor/awa2/results/" + name_add + "all_same_attr/",
                               adder="new_sent", complex_sent=True)
        end1 = time.time()
        print("Experiment2: Done with attribute number: ", str(attr_num), "in: ", end1-start1, "s")
    end2 = time.time()
    print("Total time consumption: ", end2-start, "s")


def experiment3_new_sent(module, model_name="open_clip", name_add=""):
    """Executes experiment 3 where one image gets its image attributes, while all
    other images get the same number of attributes, but their class attributes.
    This is relative similar to experiment one, but now the initially created sentences are different."""

    start = time.time()
    loadDataset = LoadDataset(model_name=model_name, new=True)
    model = loadDataset.model
    class_names = loadDataset.idx_label
    dataloader = loadDataset.data_loader
    class_idx_attribute = loadDataset.class_idx_attribute
    processor = loadDataset.processor
    # print(class_idx_attribute)
    end1 = time.time()
    print("Loaded Dataset and model in: ", end1-start, "s")
    sentenceCreator = SentenceCreator(model=model, module=module, device="cuda",
                                      processor=processor, model_name=model_name)
    # class_idx_attribute

    experiment = Experiment(model, device="cuda")
    for attr_num in [5]:  # range(0, 20):
        start1 = time.time()

        text_list = []  # 100 sentences with different randomly chosen attributes.
        for i in range(100):  # Create a selection of different sentences to choose from.
            id_sentence = sentenceCreator.create_class_sentences_with_different_attr(
                class_names, class_idx_attribute, attr_num, complex_sent=True)
            text_feat = sentenceCreator.tokenize_dict_text(text=id_sentence)
            text_list.append(text_feat)

        experiment.validation1_new(dataloader, sentenceCreator, text_list=text_list, attr_num=attr_num,
                                   file_path_base="/home/felix/new_bachelor/awa2/results/" + name_add + "attr_vs_class_attr/",
                                   adder="new_sent", complex_sent=True)
        end1 = time.time()
        print("Experiment3: Done with attribute number: ", str(attr_num), "in: ", end1-start1, "s")
    end2 = time.time()
    print("Total time consumption: ", end2-start, "s")


def experiment4_new_sent(module, model_name="open_clip", name_add=""):
    """Similar to experiment 3, but now the class labels are left out and only attributes are used."""

    start = time.time()
    loadDataset = LoadDataset(model_name=model_name, new=True)
    model = loadDataset.model
    class_names = loadDataset.idx_label
    dataloader = loadDataset.data_loader
    class_idx_attribute = loadDataset.class_idx_attribute
    processor = loadDataset.processor
    # print(class_idx_attribute)
    end1 = time.time()
    print("Loaded Dataset and model in: ", end1-start, "s")
    sentenceCreator = SentenceCreator(model=model, module=module, device="cuda",
                                      processor=processor, model_name=model_name)
    # class_idx_attribute

    experiment = Experiment(model, device="cuda")
    for attr_num in range(5, 21):
        # id_label: dict, attributes: dict, attr_num=-1
        start1 = time.time()

        text_list = []  # 100 sentences with different randomly chosen attributes.
        for i in range(100):  # Create a selection of different sentences to choose from.
            id_sentence = sentenceCreator.create_class_sentences_with_different_attr(
                class_names, class_idx_attribute, attr_num, no_label=True, complex_sent=True)
            text_feat = sentenceCreator.tokenize_dict_text(text=id_sentence)
            text_list.append(text_feat)

        experiment.validation1_new(dataloader, sentenceCreator, text_list=text_list, attr_num=attr_num, no_label=True,
                                   file_path_base="/home/felix/new_bachelor/awa2/results/" + name_add + "without_label/",
                                   adder="new_sent", complex_sent=True)
        end1 = time.time()
        print("Experiment4" + name_add + ": Done with attribute number: ", str(attr_num), "in: ", end1-start1, "s")
    end2 = time.time()
    print("Total time consumption: ", end2-start, "s")


def experiment5_new_sent(module, model_name="open_clip", name_add=""):
    """Executes experiment 3 where one image gets its image attributes, while all
    other images get the same number of attributes, but their class attributes.
    This is relative similar to experiment one, but now the initially created sentences are different."""

    start = time.time()
    loadDataset = LoadDataset(model_name=model_name, new=True)
    model = loadDataset.model
    class_names = loadDataset.idx_label
    dataloader = loadDataset.data_loader
    class_idx_attribute = loadDataset.class_idx_attribute
    processor = loadDataset.processor
    all_attributes = loadDataset.idx_attribute

    sentenceCreator = SentenceCreator(model=model, module=module, device="cuda",
                                      processor=processor, model_name=model_name)
    # class_idx_attribute

    experiment = Experiment(model, device="cuda")
    for attr_num in range(1, 21):
        start1 = time.time()
        experiment.validation3(dataloader, sentenceCreator, class_names, all_attributes=all_attributes, attr_num=attr_num,
                               file_path_base="/home/felix/new_bachelor/awa2/results/" + name_add + "one_vs_one/",
                               adder="new_sent", complex_sent=True)
        end1 = time.time()
        print("Experiment5: Done with attribute number: ", str(attr_num), "in: ", end1-start1, "s")
    end2 = time.time()
    print("Total time consumption: ", end2-start, "s")


def experiment4(module, model_name="open_clip", name_add=""):
    """Similar to experiment 3, but now the class labels are left out and only attributes are used."""

    start = time.time()
    loadDataset = LoadDataset(model_name=model_name, new=True)
    model = loadDataset.model
    class_names = loadDataset.idx_label
    dataloader = loadDataset.data_loader
    class_idx_attribute = loadDataset.class_idx_attribute
    processor = loadDataset.processor
    end1 = time.time()
    print("Loaded Dataset and model in: ", end1-start, "s")
    sentenceCreator = SentenceCreator(model=model, module=module, device="cuda",
                                      processor=processor, model_name=model_name)
    # class_idx_attribute

    experiment = Experiment(model, device="cuda")
    for attr_num in [3]:  # range(0, 21):
        # id_label: dict, attributes: dict, attr_num=-1
        start1 = time.time()

        text_list = []  # 100 sentences with different randomly chosen attributes.
        for i in range(100):  # Create a selection of different sentences to choose from.
            id_sentence = sentenceCreator.create_class_sentences_with_different_attr(
                class_names, class_idx_attribute, attr_num, no_label=True)
            text_feat = sentenceCreator.tokenize_dict_text(text=id_sentence)
            text_list.append(text_feat)
            print(id_sentence[4], "++++", id_sentence[13], "++++", id_sentence[27], "++++", id_sentence[39], "++++")
            print(class_names[4], "++++", class_names[13], "++++", class_names[27], "++++", class_names[39], "++++")

        experiment.validation1_new(dataloader, sentenceCreator, text_list=text_list, attr_num=attr_num, no_label=True,
                                   file_path_base="/home/felix/new_bachelor/awa2/results/" + name_add + "without_label/")
        end1 = time.time()
        print("Done with attribute number: ", str(attr_num), "in: ", end1-start1, "s")
    end2 = time.time()
    print("Total time consumption: ", end2-start, "s")


def experiment4_stat(module, model_name="open_clip", name_add=""):
    """It can be the case that several prompts look exactly the same, because they only use attributes that can duplicate.
        Even if the model would work perfectly it wouldn't know which of them to choose. Therefore we calculate approximately
        how many same """

    start = time.time()
    loadDataset = LoadDataset(model_name=model_name, new=True)
    model = loadDataset.model
    class_names = loadDataset.idx_label
    dataloader = loadDataset.data_loader
    class_idx_attribute = loadDataset.class_idx_attribute
    processor = loadDataset.processor
    idx_attribute = loadDataset.idx_attribute
    sentenceCreator = SentenceCreator(model=model, module=module, device="cuda",
                                      processor=processor, model_name=model_name)
    # class_idx_attribute
    infos = {"class_names": class_names, "class_attributes": class_idx_attribute, "id_attr": idx_attribute}

    experiment = Experiment(model, device="cuda")
    for attr_num in range(1, 21):
        # id_label: dict, attributes: dict, attr_num=-1
        start1 = time.time()
        experiment.validation1_stat(dataloader, sentenceCreator, infos, attr_num=attr_num, no_label=True,
                                    file_path_base="/home/felix/new_bachelor/awa2/results/" + name_add + "without_label/")
        end1 = time.time()
        print("Done with attribute number: ", str(attr_num), "in: ", end1-start1, "s")
    end2 = time.time()
    print("Total time consumption: ", end2-start, "s")


def experiment5(module, model_name="open_clip", name_add=""):
    """Executes experiment 3 where one image gets its image attributes, while all
    other images get the same number of attributes, but their class attributes.
    This is relative similar to experiment one, but now the initially created sentences are different."""

    start = time.time()
    loadDataset = LoadDataset(model_name=model_name, new=True)
    model = loadDataset.model
    class_names = loadDataset.idx_label
    dataloader = loadDataset.data_loader
    class_idx_attribute = loadDataset.class_idx_attribute
    processor = loadDataset.processor
    all_attributes = loadDataset.idx_attribute

    sentenceCreator = SentenceCreator(model=model, module=module, device="cuda",
                                      processor=processor, model_name=model_name)
    # class_idx_attribute

    experiment = Experiment(model, device="cuda")
    for attr_num in range(1, 21):
        start1 = time.time()
        experiment.validation3(dataloader, sentenceCreator, class_names, all_attributes=all_attributes, attr_num=attr_num,
                               file_path_base="/home/felix/new_bachelor/awa2/results/" + name_add + "one_vs_one/")
        end1 = time.time()
        print("Done with attribute number: ", str(attr_num), "in: ", end1-start1, "s")
    end2 = time.time()
    print("Total time consumption: ", end2-start, "s")


if __name__ == "__main__":
    # experiment1()
    # experiment2()
    # experiment4(module=clip, model_name="clip", name_add="clip/")
    # Experiment 1:
    # experiment1_new_sent()
    # experiment1_new_sent(module=clip, model_name="clip", name_add="clip/")
    # experiment1_new_sent(module="", model_name="flava", name_add="flava/")
    # Experiment 2:
    # experiment2_new_sent()
    # experiment2_new_sent(module=clip, model_name="clip", name_add="clip/")
    # experiment2_new_sent(module="", model_name="flava", name_add="flava/")
    # Experiment 3:
    # experiment3_new_sent()
    # experiment3_new_sent(module=clip, model_name="clip", name_add="clip/")
    # experiment3_new_sent(module="", model_name="flava", name_add="flava/")
    # # Experiment 4:
    # experiment4_new_sent()
    # experiment4_new_sent(module=clip, model_name="clip", name_add="clip/")
    # experiment4_new_sent(module="", model_name="flava", name_add="flava/")
    # # Experiment 5:
    # experiment5_new_sent()
    # experiment5_new_sent(module=clip, model_name="clip", name_add="clip/")
    # experiment5_new_sent(module="", model_name="flava", name_add="flava/")
    experiment3()
