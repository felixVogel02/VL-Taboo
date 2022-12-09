import pickle
import time
from cgi import test

import numpy as np
import open_clip
import torch
import torchvision.transforms as transforms
from load_dataset.cub.load_dataset import LoadDataset


class Experiment():
    """Execute the first experiment."""

    def __init__(self, model, device):
        """Initialize the class."""

        self.model = model
        self.device = device

    def save_img_encoding(self, data_loader, file_path: str = "/data/felix/new_bachelor/cub/image_embeddings/open_clip/img_embeddings.pickle"):

        # Position 0: Embedding of image with index 1, Position 1: Embedding of image with index 2.
        result = torch.zeros(11789, 512)
        for input in data_loader:
            img = input["image"]
            id = int(input["img_id"][0])
            # print("Image id: ", id)
            with torch.no_grad():
                image_features = self.model.encode_image(img.to(self.device))

            image_features /= image_features.norm(dim=-1, keepdim=True)
            # print("Shape: ", np.shape(image_features))
            result[id - 1] = image_features
        with open(file_path, "wb") as act:
            pickle.dump(result, act, protocol=pickle.HIGHEST_PROTOCOL)
        act.close()

    def save_img_encoding_flava(self, data_loader, processor, file_path: str = "/data/felix/new_bachelor/cub/image_embeddings/flava/img_embeddings.pickle"):

        # Position 0: Embedding of image with index 1, Position 1: Embedding of image with index 2.
        result = torch.zeros(11789, 768)
        trans = transforms.ToPILImage()
        for input in data_loader:
            img = trans(input["image"][0].to(self.device))
            id = int(input["img_id"][0])

            inputs = processor(images=img, return_tensors="pt", padding=True)
            with torch.no_grad():
                img_feat = self.model.get_image_features(**inputs)
            img_feat = img_feat[:, 0, :]
            img_feat = img_feat.detach().numpy()
            img_feat = torch.Tensor(img_feat / np.linalg.norm(img_feat, axis=1, keepdims=True))
            # print("Shape: ", np.shape(img_feat))

            # print("Shape: ", np.shape(image_features))
            result[id - 1] = img_feat
        with open(file_path, "wb") as act:
            pickle.dump(result, act, protocol=pickle.HIGHEST_PROTOCOL)
        act.close()


def save_open_clip_features(module=open_clip, model_name="open_clip", name_add=""):
    """Executes the first experiment where attribtues are added only to the correct sentence."""

    start = time.time()
    loadDataset = LoadDataset()
    infos = loadDataset.do_all(model_name=model_name)
    end1 = time.time()
    print("Time1: ", end1-start)

    experiment = Experiment(model=infos["model"], device="cuda")

    experiment.save_img_encoding(
        infos["data_loader"], file_path="/data/felix/new_bachelor/cub/image_embeddings/" + model_name + "/img_embeddings.pickle")

    end2 = time.time()
    print("Image encoding time consumption: ", end2-end1)
    print("Total time consumption: ", end2-start, "s")


def save_flava_features(model_name="flava"):
    """Executes the first experiment where attribtues are added only to the correct sentence."""

    start = time.time()
    loadDataset = LoadDataset()
    infos = loadDataset.do_all(model_name=model_name)
    end1 = time.time()
    print("Time1: ", end1-start)

    experiment = Experiment(model=infos["model"], device="cuda")

    experiment.save_img_encoding_flava(infos["data_loader"], processor=infos["processor"])

    end2 = time.time()
    print("Image encoding time consumption: ", end2-end1)
    print("Total time consumption: ", end2-start, "s")


if __name__ == "__main__":
    save_flava_features()
    # save_open_clip_features(module=clip, model_name="clip", name_add="")
