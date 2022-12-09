import pickle
import time

import numpy as np
import torch
import torchvision.transforms as transforms
from sentenceCreator.awa2_attributes import Attributes

from utilities.loadDataset import AttributePredictor

TOKENIZERS_PARALLELISM = True


class Save_image_labels():
    """Save labels per image for the AWA2 dataset."""

    def __init__(self, model, device):
        """Initialize the class."""

        self.model = model.to(device)
        self.device = device

    def create_image_labels(self, data_loader, processor, text_features, file_path: str = "/home/felix/new_bachelor/awa2/image_annotations/image_attr_res.pickle"):
        """Predict the attributes for every image. To the label class is added one attribute. to the other classes not."""

        # Inner dict is for recall. If class i is the label, was it also predicted. (Not other way round!)
        result = dict()

        trans = transforms.ToPILImage()
        for input in data_loader:
            img = trans(input["image"][0].to(self.device))
            label = input["label_name"]
            class_id = input["class_id"]
            attr_ids = input["attr_ids"]
            img_id = input["img_id"]

            inputs = processor(images=img, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                img_feat = self.model.get_image_features(**inputs)
            image_features = img_feat[:, 0, :].to(self.device)

            keep_text_features = []

            # Calculate similarity only for possible attributes.
            attr_id_sorted = sorted(attr_ids)
            for attr_id in attr_id_sorted:
                keep_text_features.append(text_features[attr_id.item() - 1])
            keep_text_features = torch.stack(keep_text_features).to(self.device)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ keep_text_features.T)[0]
            sim1 = similarity * 100
            sim_softmax = sim1.softmax(dim=-1)

            full_similarity = torch.zeros(85).to(self.device)
            full_raw_values = torch.zeros(85).to(self.device)

            for attr_id in range(1, 86):  # Go through all the attributes.
                if attr_id in attr_id_sorted:
                    pos = attr_id_sorted.index(attr_id)
                    full_similarity[attr_id - 1] = sim_softmax[pos]
                    full_raw_values[attr_id - 1] = similarity[pos]

            full_raw_values[full_raw_values < 0.05] = 0
            indices = torch.nonzero(full_raw_values)
            indices = torch.flatten(indices)
            indices = indices + 1

            result[img_id] = {"raw_similarities": full_raw_values, "full_similarity": full_similarity}

        with open(file_path, "wb") as act:
            pickle.dump(result, act, protocol=pickle.HIGHEST_PROTOCOL)
        act.close()


def main(model_name="flava"):

    start = time.time()
    attrPred = AttributePredictor(model_name=model_name)
    attr_converter_class = Attributes()

    folder = "opposite4_attr_"

    raw_text = attrPred.create_complex_text_labels(attrPred.idx_attribute, attr_converter_class.attribute_to_text)
    text_features = attrPred.tokenize_text_flava(raw_text)
    print("Shape of the text: ", np.shape(text_features))

    end1 = time.time()
    print("Text tokenized in: ", end1-start, "s")

    data_loader, dataset, model, processor = attrPred.load_model(
        attrPred.label_idx, attrPred.class_idx_attribute, attrPred.class_idx_attribute_idx, model_name=model_name, device="cuda")

    end2 = time.time()
    print("Loaded model in: ", end2-end1, "s")
    save_img_labels = Save_image_labels(model, device="cuda")
    save_img_labels.create_image_labels(
        data_loader, processor, text_features, file_path="/home/felix/new_bachelor/awa2/image_annotations/flava/image_attr_res.pickle")

    end = time.time()
    print("Time consumption: ", end - start)


def after_care():
    torch.cuda.empty_cache()
    # file_path = "/home/felix/new_bachelor/awa2/image_annotations/image_attr_res.pickle"
    # with open(file_path, "rb") as act:
    #     result = pickle.load(act)
    # act.close()
    # new_res = {key: result[key]["raw_similarities"] for key in result.keys()}

    file_path = "/home/felix/new_bachelor/awa2/image_annotations/flava/image_attr_res.pickle"
    #file_path = "/home/felix/new_bachelor/awa2/image_annotations/image_attr_res.pickle"
    with open(file_path, "rb") as act:
        result = pickle.load(act)
    act.close()

    new_res = dict()
    avg = 0
    cnt = 0
    for key in result.keys():
        cnt += 1
        sim = result[key]["raw_similarities"]
        # open_clip: cut_off = 0.2 -> 19.67034 average number of attributes.
        # clip: cut_off = 0.24 -> 20.61462 average number of attributes.
        # flava. cut_off = 0.245 -> 20.104818 average number of attributes.
        sim[sim < 0.245] = 0
        indices = torch.nonzero(sim)
        indices = torch.flatten(indices)
        indices = indices + 1  # Because indices of the list start with 0, but attribtue indices with 1.
        new_res[key] = indices
        avg += len(indices)
    print("Average number of attributes: ", avg / cnt)

    file_path = "/home/felix/new_bachelor/awa2/image_annotations/flava/image_attr_raw_similarities_indices.pickle"
    with open(file_path, "wb") as act:
        pickle.dump(new_res, act, protocol=pickle.HIGHEST_PROTOCOL)
    act.close()


def after_care2():
    file_path = "/home/felix/new_bachelor/awa2/image_annotations/flava/image_attr_raw_similarities_indices.pickle"
    with open(file_path, "rb") as act:
        new_res = pickle.load(act)
    act.close()
    new_one = dict()
    for key in new_res.keys():
        if type(key) == int:
            new_one[key] = new_res[key]
        else:
            new_one[key.item()] = new_res[key]

    with open(file_path, "wb") as act:
        pickle.dump(new_one, act, protocol=pickle.HIGHEST_PROTOCOL)
    act.close()


if __name__ == "__main__":
    main(model_name="clip")
    after_care()
    after_care2()
