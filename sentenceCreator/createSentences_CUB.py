import random

import clip
import open_clip
import torch

TOKENIZERS_PARRALLELISM = True


class SentenceCreator():
    """Creates sentences."""

    def __init__(self, model=False, module=open_clip, device="cuda", processor=None, model_name=""):
        """Initializes the class."""

        self.model = model.to(device) if model else None
        self.module = module
        self.device = device
        self.processor = processor
        self.model_name = model_name

    def create_class_sentences(self, id_label: dict) -> dict:
        """From a given dict it creates for each label a sentence out of it."""

        id_sentence = dict()
        for id in id_label.keys():
            label = id_label[id]
            sentence = "a photo of a " + label + "."
            id_sentence[int(id)] = sentence
        return id_sentence

    def create_class_sentences_with_same_attr(self, id_label: dict, attributes: dict, attr_num=-1) -> dict:
        """From a given dict it creates for each label a sentence out of it."""

        if attr_num == -1:
            chosed_attr = attributes.keys()
        else:
            chosed_attr = random.sample(attributes.keys(), attr_num)

        id_sentence = dict()
        for id in id_label.keys():
            label = id_label[id]
            sentence = "a photo of a " + label
            if len(chosed_attr) > 0:
                for attr_desc in chosed_attr:
                    attrs = attributes[attr_desc]  # attrs should only contain one attribute
                    sentence += " that " + attr_desc
                    if len(attrs) > 1:  # Ok or not?! Not!
                        print("Error! Only one attribute per description should be assigned! With attr_descr:",
                              attr_desc, "Attributes: ", attrs)
                    # Only use one attribute to go along with experiment 3.
                    sentence += " " + attrs[0] + ","
                    # for attr in attrs:
                    #     sentence += " " + attr + ","
                sentence = sentence[:-1] + "."
            else:
                sentence += "."
            id_sentence[int(id)] = sentence
        return id_sentence

    def create_class_sentences_with_different_attr(self, id_label: dict, attributes: dict, id_attr: dict, attr_num=-1, no_label=False, only_chosed_attr=False, ignore_label="") -> dict:
        """From a given dict it creates for each label a sentence out of it."""

        id_sentence = dict()
        all_attr = []
        for id in id_label.keys():
            if attr_num == -1:
                choosed_attr = attributes[int(id)]
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
            if no_label:
                sentence = "a photo of a bird"
            else:
                sentence = "a photo of a " + label
            all_attr_desc = []
            for attr_id in chosed_attr:
                attr_id = str(attr_id)
                # print("Attributes: ", id_attr[attr_id])
                attr_desc = id_attr[attr_id][0]
                if attr_desc in all_attr_desc:
                    print("Attr_desc: ", attr_desc, " has already been used!!!")
                all_attr_desc.append(attr_desc)
                attrs = id_attr[attr_id][1]  # attrs should only contain one attribute
                if type(attrs) == str:
                    attrs = [attrs]
                sentence += " that " + attr_desc
                for attr in attrs:
                    sentence += " " + attr + ","
            if len(chosed_attr) > 0:
                sentence = sentence[:-1] + "."
            else:
                sentence += "."
            id_sentence[int(id)] = sentence
        if only_chosed_attr:
            return all_attr
        return id_sentence

    def create_class_sentences_with_attributes(self, label: str, attributes: dict, attr_num=-1, no_label=False) -> dict:
        """From a given dict it creates for each label a sentence out of it."""

        if attr_num == -1:
            chosed_attr = attributes.keys()
        else:
            chosed_attr = random.sample(attributes.keys(), attr_num)
        if no_label:
            sentence = "a photo of a bird"
        else:
            sentence = "a photo of a " + label
        if len(chosed_attr) > 0:
            for attr_desc in chosed_attr:
                attrs = attributes[attr_desc]  # attrs should only contain one attribute
                sentence += " that " + attr_desc
                for attr in attrs:
                    sentence += " " + attr + ","
            sentence = sentence[:-1] + "."
        else:
            sentence += "."

        return sentence

    def create_class_sentences_with_attributes_with_replacer(self, attributes: dict, replacer="zzzzzzz", attr_num=-1, no_label=False) -> dict:
        """With this function afterwards jsut the label can be exchanged while the whole rest of the sentence stays the same."""

        if attr_num == -1:
            chosed_attr = attributes.keys()
        else:
            chosed_attr = random.sample(attributes.keys(), attr_num)
        if no_label:
            sentence = "a photo of a bird"
        else:
            sentence = "a photo of a " + replacer
        if len(chosed_attr) > 0:
            for attr_desc in chosed_attr:
                attrs = attributes[attr_desc]  # attrs should only contain one attribute
                sentence += " that " + attr_desc
                for attr in attrs:
                    sentence += " " + attr + ","
            sentence = sentence[:-1] + "."
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
