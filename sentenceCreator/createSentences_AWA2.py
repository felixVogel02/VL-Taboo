import random

import clip
import open_clip
import torch


class SentenceCreator():
    """Creates sentences."""

    def __init__(self, model, module=open_clip, device="cuda", processor=None, model_name=""):
        """Initializes the class."""

        self.model = model.to(device)
        self.module = module
        self.device = device
        self.processor = processor
        self.model_name = model_name
        # Paper groups.
        # self.attr1 = ["group"]  # no
        # self.attr_size = ["big", "small"]  # Size
        # self.attr_shape = ["furry", "hairless",
        #                    "toughskin", "bulbous", "lean", "bipedal", "quadrapedal"]  # Shape
        # self.attr_color = ["black", "white", "blue", "brown", "gray", "orange",
        #                    "red", "yellow"]  # Color
        # self.attr_rest1 = ["fierce", "timid", "smart", "solitary", "domestic", "weak", "strong", "smelly"]  # no
        # self.attr_rest2 = ["active", "inactive", "nocturnal"]  # no

        # self.places = ["arctic", "coastal", "desert", "bush", "plains", "forest", "fields",
        #                "jungle", "mountains", "ocean", "ground", "water", "tree", "cave"]

        # self.nouns = ["fish", "meat", "plankton", "vegetation", "insects", "forager", "grazer",
        #               "scavenger", "skimmer", "stalker", "nestspot", "hunter"]  # no

        # self.nouns_rest = ["newworld", "oldworld"]  # no

        # self.add_nouns = ["flippers", "hands", "hooves", "pads", "paws", "longleg", "longneck", "patches", "spots", "stripes",
        #                   "tail", "chewteeth", "meatteeth", "buckteeth", "strainteeth", "horns", "claws", "tusks", "agility", "muscle"]

        # self.verbs = ["flys", "hops", "swims", "tunnels", "walks", "hibernate"]  # no
        # self.verb_attr = ["fast", "slow"]  # no
        self.quantity = ["group"]  # Should "solitary" be moved to here?!
        self.size = ["big", "small"]
        self.movement_verbs = ["flys", "hops", "swims", "tunnels", "walks"]
        self.movement_adj = ["fast", "slow"]
        self.color = ["black", "white", "blue", "brown", "gray", "orange",
                      "red", "yellow"]  # Color
        self.activity = ["active", "inactive", "nocturnal"]
        self.shape_adj = ["furry", "hairless", "toughskin", "bulbous", "lean",  # smelly added
                          "bipedal", "quadrapedal", "weak", "strong", "smelly"]  # furry=pelzig fierce=wild, timid=schüchtern, solitary=einsam
        self.shape_nouns = ["flippers", "hands", "hooves", "pads", "paws", "longleg", "longneck", "patches", "spots", "stripes",
                            "tail", "chewteeth", "meatteeth", "buckteeth", "strainteeth", "horns", "claws", "tusks", "agility", "muscle"]
        # forager=Sammler, grazer=Graser (weidetier), scavenger=Assfresser
        self.food = ["fish", "meat", "plankton", "vegetation", "insects"]
        self.eater_description = ["forager", "grazer", "scavenger", "skimmer", "stalker", "hunter"]
        self.places = ["arctic", "coastal", "desert", "bush", "plains", "forest", "fields",
                       "jungle", "mountains", "ocean", "ground", "water", "tree", "cave", "newworld", "oldworld"]
        self.behaviour = ["fierce", "timid", "smart", "solitary", "domestic"]
        # no, oldworld: Asia, Africa, Europe, newworld: America, North America, South America, Central America
        self.characteristics = ["hibernate", "nestspot"]  # hibernate=überwintern

    def create_class_sentences(self, id_label: dict) -> dict:
        """From a given dict it creates for each label a sentence out of it."""

        id_sentence = dict()
        for id in id_label.keys():
            label = id_label[id].replace("+", " ")
            sentence = "a photo of a " + label + "."
            id_sentence[int(id)] = sentence
        return id_sentence

    def create_class_sentences_with_same_attr(self, id_label: dict, attributes: dict, attr_num=-1, complex_sent=False) -> dict:
        """From a given dict it creates for each label a sentence out of it."""

        if attr_num == -1:
            chosed_attr = attributes.keys()
        else:
            chosed_attr = random.sample(attributes.keys(), attr_num)

        id_sentence = dict()
        for id in id_label.keys():
            label = id_label[id]
            attr_list = []
            for attr_id in chosed_attr:
                attr_list.append(attributes[attr_id])
            if complex_sent:
                sentence = self.create_sentences(attr_list, label=label)
            else:
                sentence = "a photo of a " + label.replace("+", " ")
                if len(chosed_attr) > 0:
                    sentence += " with attributes "
                    for attr_id in chosed_attr:
                        attr = attributes[attr_id]  # attrs should only contain one attribute
                        sentence += attr + ", "
                    sentence = sentence[:-2] + "."
                else:
                    sentence += "."
            id_sentence[int(id)] = sentence
        return id_sentence

    def create_class_sentences_with_different_attr(self, id_label: dict, attributes: dict, attr_num=-1, no_label=False,
                                                   only_chosed_attr=False, ignore_label="", complex_sent=False) -> dict:
        """From a given dict it creates for each label a sentence out of it."""

        id_sentence = dict()
        all_attr = []
        for id in id_label.keys():
            if attr_num == -1:
                chosed_attr = attributes[int(id)]
            else:
                chosed_attr = random.sample(attributes[int(id)], attr_num)

            label = id_label[id]
            if only_chosed_attr:
                if label == ignore_label:
                    # For stats. For the correct label we know that it has the
                    # correct attribtues inherent. (At least mostly because of the difference between class and image labels.)
                    continue
                all_attr.append(chosed_attr)
                continue

            if complex_sent:
                if no_label:
                    sentence = self.create_sentences(chosed_attr, label=False)
                else:
                    sentence = self.create_sentences(chosed_attr, label=label)
            else:
                if no_label:
                    sentence = "a photo of an animal"
                else:
                    sentence = "a photo of a " + label.replace("+", " ")
                if len(chosed_attr) > 0:
                    sentence += " with attributes "
                    for attr in chosed_attr:
                        sentence += attr + ", "
                    sentence = sentence[:-2] + "."
                else:
                    sentence += "."

            id_sentence[int(id)] = sentence
        if only_chosed_attr:
            return all_attr
        return id_sentence

    def create_class_sentences_with_attributes(self, label: str, attributes: dict, attr_num=-1, no_label=False, complex_sent=False) -> dict:
        """From a given dict it creates for each label a sentence out of it."""

        if attr_num == -1:
            chosed_attr = attributes.keys()
        else:
            chosed_attr = random.sample(attributes.keys(), attr_num)
        if complex_sent:
            attrs = [attributes[attr_id] for attr_id in chosed_attr]
            if no_label:
                sentence = ""
                for i in attrs:
                    sentence += i + " "
                #sentence = self.create_sentences(attrs, label=False)
            else:
                sentence = self.create_sentences(attrs, label=label)
        else:
            if no_label:
                sentence = "a photo of an animal"
            else:
                sentence = "a photo of a " + label.replace("+", " ")
            if len(chosed_attr) > 0:
                sentence += " with attributes "
                for attr_id in chosed_attr:
                    attr = attributes[attr_id]  # attrs should only contain one attribute
                    sentence += attr + ", "
                sentence = sentence[:-2] + "."
            else:
                sentence += "."

        return sentence

    def create_sentences(self, attributes, label=False):
        """Create sentences with specified number of attributes."""

        attr_set = set(attributes)
        # Create the sentence.
        text = "a photo of a "
        for attr in attr_set:
            if attr in self.quantity:
                text += attr + " of "  # "group of "
        for attr in attr_set:
            if attr in self.size:
                text += attr + ", "  # Size
        for attr in attr_set:
            if attr in self.shape_adj:
                text += attr + ", "  # Shape
        for attr in attr_set:
            if attr in self.color:
                text += attr + ", "  # Color
        for attr in attr_set:
            if attr in self.behaviour:
                text += attr + ", "  # rest1.
        for attr in attr_set:
            if attr in self.activity:
                text += attr + ", "  # rest2.

        if text[-2] == ",":  # Remove comma at the end, if it is there.
            text = text[:-2] + " "

        if label:
            text += label + " "
        else:
            text += "animal "
        text_add = ""
        added_since_label = False
        for attr in attr_set:
            if attr in self.shape_nouns:
                text_add += attr + ", "  # Nouns
        if len(text_add) > 0:
            text += "with "
            text += text_add[:-2] + " "

        text_add = ""
        for attr in attr_set:
            if attr in self.movement_verbs:
                text_add += attr + ", "
        if len(text_add) > 0:
            text += "that " + text_add[:-2] + " "
            added_since_label = True
        text_add1 = ""
        for attr in attr_set:
            if attr in self.movement_adj:
                text_add1 += attr + " or "
        if len(text_add1) > 0:
            if len(text_add) == 0:
                text += "that is "
                added_since_label = True
            text += text_add1[:-3]

        text_add = ""
        for attr in attr_set:
            if attr in self.characteristics:
                text_add += attr + ", "
        if len(text_add) > 0:
            text += "that " + text_add[:-2] + " "
            added_since_label = True

        text_add = ""
        for attr in attr_set:
            if attr in self.eater_description:
                text_add += attr + ", "
        if len(text_add) > 0:
            text += "that is a " + text_add[:-2] + " "
            added_since_label = True

        text_add = ""
        for attr in attr_set:
            if attr in self.food:
                text_add += attr + ", "
        if len(text_add) > 0:
            if added_since_label:
                text += "and eats " + text_add[:-2] + " "
            else:
                text += "that eats " + text_add[:-2] + " "
            added_since_label = True

        text_add = ""
        for attr in attr_set:
            if attr in self.places:
                text_add += attr + " or "
        if len(text_add) > 0:
            if added_since_label:
                text += "and is in " + text_add[:-4]
            else:
                text += "that is in " + text_add[:-4]
        if text[-1] == " ":
            text = text[:-1]
        text += "."

        text = text.replace("+", " ")  # Because multi word labels are seperated by a +"+".
        return text

    def create_sentences_paper(self, attributes, label=False):
        """Create sentences with specified number of attributes."""

        attr_set = set(attributes)
        # Create the sentence.
        text = "a photo of a "
        for attr in attr_set:
            if attr in self.attr1:
                text += attr + " of "  # "group of "
        for attr in attr_set:
            if attr in self.attr_size:
                text += attr + ", "  # Size
        for attr in attr_set:
            if attr in self.attr_shape:
                text += attr + ", "  # Shape
        for attr in attr_set:
            if attr in self.attr_color:
                text += attr + ", "  # Color
        for attr in attr_set:
            if attr in self.attr_rest1:
                text += attr + ", "  # rest1.
        for attr in attr_set:
            if attr in self.attr_rest2:
                text += attr + ", "  # rest2.

        if text[-2] == ",":  # Remove comma at the end, if it is there.
            text = text[:-2] + " "

        if label:
            text += label + " "
        else:
            text += "animal "
        text_add = ""
        added_since_label = False
        for attr in attr_set:
            if attr in self.nouns:
                text_add += attr + ", "  # Nouns
        if len(text_add) > 0:
            if label:
                text += "with "
            else:
                text += " "
            text += text_add[:-2] + " "

        text_add = ""
        for attr in attr_set:
            if attr in self.verbs:
                text_add += attr + ", "
        if len(text_add) > 0:
            text += "that " + text_add[:-2] + " "
            added_since_label = True
        text_add1 = ""
        for attr in attr_set:
            if attr in self.verb_attr:
                text_add1 += attr + " or "
        if len(text_add1) > 0:
            if len(text_add) == 0:
                text += "that is "
                added_since_label = True
            text += text_add1[:-3]

        text_add = ""
        for attr in attr_set:
            if attr in self.add_nouns:
                text_add += attr + ", "
        if len(text_add) > 0:
            text += "that has " + text_add[:-2] + " "
            added_since_label = True

        text_add = ""
        for attr in attr_set:
            if attr in self.places:
                text_add += attr + " or "
        for attr in attr_set:
            if attr in self.nouns_rest:
                text_add += attr + " or "
        if len(text_add) > 0:
            if added_since_label:
                text += "and is in " + text_add[:-4]
            else:
                text += "that is in " + text_add[:-4]
        if text[-1] == " ":
            text = text[:-1]
        text += "."

        text = text.replace("+", " ")  # Because multi word labels are seperated by a +"+".
        return text

    def create_class_sentences_with_attributes_with_replacer(self, attributes: dict, attr_num=-1, replacer="zzzzzzz", no_label=False) -> dict:
        """From a given dict it creates for each label a sentence out of it."""

        if attr_num == -1:
            chosed_attr = attributes.keys()
        else:
            chosed_attr = random.sample(attributes.keys(), attr_num)
        if no_label:
            sentence = "a photo of an animal"
        else:
            sentence = "a photo of a " + replacer
        if len(chosed_attr) > 0:
            sentence += " with attributes "
            for attr_id in chosed_attr:
                attr = attributes[attr_id]  # attrs should only contain one attribute
                sentence += attr + ", "
            sentence = sentence[:-2] + "."
        else:
            sentence += "."

        return sentence

    def tokenize_single_sentence(self, sentence):
        """Tokenize a single sentence."""

        if self.model_name == "flava":
            with torch.no_grad():
                inputs = self.processor(text=[sentence], return_tensors="pt", padding=True).to(self.device)

                with torch.no_grad():
                    txt_feat = self.model.get_text_features(**inputs)
                txt_feat = txt_feat[:, 0, :]
                # txt_feat = txt_feat.detach().numpy()
                # txt_feat = torch.Tensor(txt_feat / np.linalg.norm(txt_feat, axis=1, keepdims=True))

        else:
            with torch.no_grad():
                texts = self.module.tokenize([sentence]).to(self.device)  # tokenize
                # print("Tokenized single sentence: ", texts)
                txt_feat = self.model.encode_text(texts)
                # print("Old class embedding: ", class_embeddings)
        return txt_feat

    def tokenize_dict_text(self, text: dict):
        """Tokenizes the sentences in the dict in the correct order of the keys and tokenizes the sentences."""

        if self.model_name == "flava":
            with torch.no_grad():
                texts = []
                for pos in sorted(text.keys()):
                    texts.append(text[pos])

                inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)

                with torch.no_grad():
                    txt_feat = self.model.get_text_features(**inputs)
                txt_feat = txt_feat[:, 0, :]
                # txt_feat = txt_feat.detach().numpy()
                # txt_feat = torch.Tensor(txt_feat / np.linalg.norm(txt_feat, axis=1, keepdims=True))

        else:
            with torch.no_grad():
                texts = []
                for pos in sorted(text.keys()):
                    texts.append(text[pos])

                texts = self.module.tokenize(texts).to(self.device)  # tokenize
                # print("Tokenized:", texts)
                txt_feat = self.model.encode_text(texts)
        return txt_feat

    def norm_text(self, text_features):
        """Normalize the text."""

        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features
