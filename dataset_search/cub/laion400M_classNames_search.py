import sys

sys.path.append("/home/felix/new_bachelor/awa2/")  # noqa
sys.path.append("/home/felix/new_bachelor/cub/")  # noqa

import copy
import os
import pickle
import random
from cgi import test
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import seaborn as sn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import utilities
from ahocorapy.keywordtree import KeywordTree
from fastparquet import ParquetFile
from load_dataset import LoadDataset
from sklearn import metrics
from sklearn.metrics import confusion_matrix


def create_search_tree(attr_idx_label):
    """Build a search tree."""

    new_labels_idx = dict()  # Maps from the labels of each class to the index of the class.
    kwtree = KeywordTree(case_insensitive=True)  # We don't care about the case.
    # To avoid recognizing a word that is only a subword. But still keep plural, ...
    label_suffix = [".", "!", "?", ":", " ", ",", ";", "-",
                    "s.", "s!", "s?", "s:", "s ", "s,", "s;", "s-",
                    "es.", "es!", "es?", "es:", "es ", "es,", "es;", "es-"]
    # for attr_idx in attr_idx_attr_list_dict.keys():
    for attr_idx in attr_idx_label.keys():  # Different versions/declinations of the same word.
        label_list = attr_idx_label[attr_idx]
        if type(attr_idx_label[attr_idx]) != list:  # Make it a list.
            label_list = [attr_idx_label[attr_idx]]

        for label in label_list:
            for sfx in label_suffix:
                label1 = " " + label + sfx
                kwtree.add(label1)  # Doesn't work with regex.
                if label1 not in new_labels_idx.keys():
                    new_labels_idx[label1] = attr_idx
                else:
                    print("Error, should not happen")
                    # new_labels_idx[label1].append(attr_idx)
    kwtree.finalize()
    return kwtree, new_labels_idx


def get_names(adder=""):
    """Return all the names of the Laion400M text files."""

    # Search in all text data of LAION400M after the labels:
    names_in = []
    names_out = []
    for i in range(0, 32):
        if i < 10:
            numb = "0" + str(i)
        else:  # For 2 digit numbers.
            numb = str(i)
        name_in = "/data/felix/Laion400M/part-000" + numb + "-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet"
        name_out = "/data/felix/new_bachelor/cub/dataset_analysis/laion400M_search/" + adder + "result_" + numb + ".pickle"
        names_in.append(name_in)
        names_out.append(name_out)

    return names_in, names_out


def count_sampels(names_in, names_out, kwtree, new_labels_idx):
    """Count the samples."""

    for name_idx in range(len(names_in)):
        name_in = names_in[name_idx]
        name_out = names_out[name_idx]

        pf = ParquetFile(name_in)
        dff = pf.to_pandas()["TEXT"]
        cleared_dff = dff.dropna()  # Samples with Nan as text should not be counted.

        # Calculate occurences per file:
        result_mtx = np.zeros((len(cleared_dff), 201), dtype=bool)  # list()
        cnt = 0
        for index, value in cleared_dff.items():  # [:int(len(dff) / 10)]
            # Select important text from the line.
            text = " " + value + " "
            # To have a whitespace at the beginning and at the end.
            # Check class occurances in the text:
            result = kwtree.search_all(text)
            for found in result:
                # print("Count: ", cnt, "--------", found)
                keyword = found[0]
                pos = new_labels_idx[keyword]
                # Don't count occurences. If there are several same occurences, still 1 should be saved.
                result_mtx[cnt][pos] = True
                # print(pos, row[pos])
            cnt += 1
        with open(name_out, "wb") as act:
            pickle.dump(result_mtx, act, protocol=pickle.HIGHEST_PROTOCOL)
        act.close()
        print("Done file number: ", name_idx)


def get_occurrence(names_out, idx_label):
    """Save the last word occurrences of each class."""

    # Calculate relative occurence of the classes in the english training samples.
    samples = 0
    occurence = np.zeros(len(idx_label.keys()) + 1)
    a = np.shape(occurence)
    for name_out in names_out:
        with open(name_out, "rb") as act:
            result_mtx = pickle.load(act)
            b = np.shape(result_mtx)
        act.close()
        samples += np.shape(result_mtx)[0]
        # print(samples)
        # print(len(np.sum(result_mtx, axis=0)))
        occurence += np.sum(result_mtx, axis=0)
        print("One step done.")
    # occurence = occurence / samples

    print("Samples: ", samples)
    print("Occurences: ", occurence)
    return occurence


def main():
    loadDataset = LoadDataset()
    infos = loadDataset.give_dataset_infos()
    idx_label_old = infos["class_names"]
    idx_label = dict()
    for key in idx_label_old.keys():
        idx_label[int(key)] = idx_label_old[key]

    kwtree, new_labels_idx = create_search_tree(idx_label)

    with open("/data/felix/new_bachelor/cub/dataset_analysis/laion400M_search/class_names/aho_corasick_class_repr_only_ending.pickle", "wb") as act:
        pickle.dump(kwtree, act, protocol=pickle.HIGHEST_PROTOCOL)
    act.close()
    with open("/data/felix/new_bachelor/cub/dataset_analysis/laion400M_search/class_names/new_labels_idx_only_ending.pickle", "wb") as act:
        pickle.dump(new_labels_idx, act, protocol=pickle.HIGHEST_PROTOCOL)
    act.close()

    names_in, names_out = get_names()
    count_sampels(names_in, names_out, kwtree, new_labels_idx)
    occurrence = get_occurrence(names_out, idx_label)
    with open("/home/felix/new_bachelor/cub/dataset_analysis/laion400M_search/class_names/absolute_occurence.pickle", "wb") as act:
        pickle.dump(occurrence, act, protocol=pickle.HIGHEST_PROTOCOL)
    act.close()


def end_of_main():
    loadDataset = LoadDataset()
    infos = loadDataset.give_dataset_infos()
    idx_label_old = infos["class_names"]
    idx_label = dict()
    for key in idx_label_old.keys():
        idx_label[int(key)] = idx_label_old[key]
    names_in, names_out = get_names()

    occurrence = get_occurrence(names_out, idx_label)
    with open("/home/felix/new_bachelor/cub/dataset_analysis/laion400M_search/class_names/absolute_occurence.pickle", "wb") as act:
        pickle.dump(occurrence, act, protocol=pickle.HIGHEST_PROTOCOL)
    act.close()


def save_attr_occ_mapping():
    """Save a mapping from the attribute index or name to it's nubmer of occurrences."""

    loadDataset = LoadDataset()
    infos = loadDataset.give_dataset_infos()
    idx_label_old = infos["class_names"]
    idx_label = dict()
    for key in idx_label_old.keys():
        idx_label[int(key)] = idx_label_old[key]

    with open("/home/felix/new_bachelor/cub/dataset_analysis/laion400M_search/class_names/absolute_occurence.pickle", "rb") as act:
        occurrence = pickle.load(act)
    act.close()
    res_mapping = {attr_idx: 0 for attr_idx in idx_label.keys()}
    res_mapping_readable = {attr: 0 for attr in idx_label.values()}
    for i in idx_label.keys():
        # Because idx_attribtue keys starts with 1, but occurrence at index 0 chould be 0 (no entries).
        occ = occurrence[i]
        res_mapping[i] = occ
        res_mapping_readable[idx_label[i]] = occ

    with open("/home/felix/new_bachelor/cub/dataset_analysis/laion400M_search/class_names/attr_idx_occurence.pickle", "wb") as act:
        pickle.dump(res_mapping, act, protocol=pickle.HIGHEST_PROTOCOL)
    act.close()
    with open("/home/felix/new_bachelor/cub/dataset_analysis/laion400M_search/class_names/attr_readable_occurence.pickle", "wb") as act:
        pickle.dump(res_mapping_readable, act, protocol=pickle.HIGHEST_PROTOCOL)
    act.close()


if __name__ == "__main__":
    # main()
    end_of_main()
    save_attr_occ_mapping()
